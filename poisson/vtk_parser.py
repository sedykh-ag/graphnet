import xml.etree.ElementTree as ET
import numpy as np
from absl import app
from absl import flags
from pathlib import Path
from enum import Enum
from collections import namedtuple

FLAGS = flags.FLAGS
flags.DEFINE_string("dir", None, "Input XML file directory.")
flags.mark_flag_as_required("dir")

Mesh = namedtuple('Mesh', ['mesh_pos', 'connectivity', 'scalar_field'])
dtypes = {
    "Int8"   : int,
    "UInt8"  : int,
    "Int16"  : int,
    "UInt16" : int,
    "Int32"  : int,
    "UInt32" : int,
    "Int64"  : int,
    "UInt64" : int,
    "Float32": float,
    "Float64": float}

def parse_string(x, dtype, noc):
    arr = np.fromstring(x, sep=" ", dtype=dtype)
    return arr.reshape(-1, noc)

def parse_vtu(ds_path):
    """ .vtu file parsing """
    mesh_pos = None
    connectivity = None
    scalar_field = None
    
    tree = ET.parse(ds_path)
    root = tree.getroot()

    # Points
    for points in root.iter("Points"):
        dataArray = points.find("DataArray")
        mesh_pos = parse_string(dataArray.text, dtype=dtypes[dataArray.attrib["type"]], noc=3)
    # Cells
    for cells in root.iter("Cells"):
        for dataArray in cells.iter("DataArray"):
            if dataArray.attrib["Name"] == "connectivity":
                connectivity = parse_string(dataArray.text, dtype=dtypes[dataArray.attrib["type"]], noc=3)
    # Fields
    for pointData in root.iter("PointData"):
        for dataArray in pointData.iter("DataArray"):
            scalar_field = parse_string(dataArray.text, dtype=dtypes[dataArray.attrib["type"]], noc=1)
    
    return Mesh(mesh_pos, connectivity, scalar_field)
    
def main(argv):
    del argv
    
    # get .vtu file path from .pvd file
    pvd_path = Path(FLAGS.dir)
    root = ET.parse(pvd_path).getroot()
    vtu_path = None
    for dataSet in root.iter("DataSet"):
        vtu_path = pvd_path.parent / dataSet.get("file")
    
    # parse vtu file
    mesh = parse_vtu(vtu_path)
    print(mesh.scalar_field.shape)
    
    
if __name__ == "__main__":
    app.run(main)
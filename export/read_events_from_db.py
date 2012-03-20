#!/usr/bin/env python
#coding=utf-8

import sys
import tables

cell_db = sys.argv[1]
output = sys.argv[2]

if __name__ == "__main__":

    h5f = tables.openFile(cell_db)
    with open(output, 'w') as f:
        for node in h5f.walkNodes():
            if node._v_name == 'events':
                path = "/".join(node._v_pathname.split('/')[:-2])
                data = " ".join(map(str,node.read()))
                f.write(path + " " + data + '\n')


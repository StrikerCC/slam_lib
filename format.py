import numpy as np
import time
import xml.etree.ElementTree


def pt_3d_format(pts_3d):
    pass


def pts_2d_format(pts_2d):
    pass


def pts_2d_2_3d_homo(pts_2d):
    assert pts_2d.shape[1] == 2
    return np.concatenate([np.copy(pts_2d), np.ones((pts_2d.shape[0], 1))], axis=-1)


class timer:
    def __init__(self):
        self.time_accounting = {}

    def start(self, name=''):
        name = str(name)
        self.time_accounting[name] = time.time()

    def end(self, name=''):
        name = str(name)
        self.time_accounting[name] = time.time() - self.time_accounting[name]

    def __str__(self):
        spacing = '     '
        output = 'time accounting:\n'
        for name in self.time_accounting.keys():
            output += spacing + str(name) + ' : ' + str(self.time_accounting[name]) + '\n'
        return output


def test_timer():
    t = timer()

    t.start(3)
    t.start(1)
    for i in range(1000000): pass
    t.end(1)

    t.start(2)
    for i in range(1000000): pass
    t.end(3)

    for i in range(10000):
        pass
    t.end(2)

    print(t)


def xml_2_pts(xml_file_path):
    pts_xml = xml.etree.ElementTree.parse(xml_file_path)
    root = pts_xml.getroot()
    pts = []
    for i in range(len(root)):
        pt_dict = root[i].attrib
        print(root[i].tag)
        print(pt_dict)
        pts.append([float(pt_dict['x']), float(pt_dict['y'])])
    return pts


def pts_2_xml(pts, xml_file_path):
    if isinstance(pts, np.ndarray):
        pts = pts.tolist()
    vector_xml = xml.etree.ElementTree.Element('vector')
    for i in range(len(pts)):
        pt = pts[i]
        assert len(pt) == 2
        pt_tag = 'point_'+str(i)
        pt_xml = {'x': '{:.6f}'.format(pt[0]), 'y': '{:.6f}'.format(pt[1])}
        xml.etree.ElementTree.SubElement(vector_xml, pt_tag, attrib=pt_xml)
    xml_file = xml.etree.ElementTree.tostring(vector_xml)
    with open(xml_file_path, 'wb') as f:
        f.write(xml_file)
    return True


def main():
    xml_file_path = '/home/cheng/Downloads/Vectors.xml'
    pts = xml_2_pts(xml_file_path)
    pts_2_xml(pts, '/home/cheng/Downloads/Vectors_Cheng.xml')
    test_timer()


if __name__ == '__main__':
    main()

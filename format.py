import os.path

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


def xml_2_rois(xml_file_path):
    if not os.path.isfile(xml_file_path):
        return None
    pts_xml = xml.etree.ElementTree.parse(xml_file_path)
    root = pts_xml.getroot()
    vectors_xml = root
    rois = []
    for i_roi in range(len(vectors_xml)):
        vector_dict = vectors_xml[i_roi]
        pts = []
        for i_pt in range(len(vector_dict)):
            pt_dict = vector_dict[i_pt].attrib
            pts.append([float(pt_dict['x']), float(pt_dict['y'])])
        rois.append(pts)
    return rois


def rois_2_xml(rois, xml_file_path):
    print('writing to ', xml_file_path)
    # if rois is None or len(rois) == 0:
    #     if os.path.isfile(xml_file_path):
    #         os.remove(xml_file_path)
    #     return True
    if isinstance(rois, np.ndarray):
        rois = rois.tolist()
    vectors_xml = xml.etree.ElementTree.Element('vectors')
    num_pt = 0
    for i_roi, roi in enumerate(rois):
        vector_tag = 'vector_'+str(i_roi)
        pts = roi
        if len(pts) > 0:
            vector_xml = xml.etree.ElementTree.SubElement(vectors_xml, vector_tag)
            for i_pt, pt in enumerate(pts):
                assert len(pt) == 2
                pt_tag = 'point_'+str(num_pt)
                pt_xml = {'x': '{:.6f}'.format(pt[0]), 'y': '{:.6f}'.format(pt[1])}
                xml.etree.ElementTree.SubElement(vector_xml, pt_tag, attrib=pt_xml)
                num_pt += 1

    xml_file = xml.etree.ElementTree.tostring(vectors_xml)

    if os.path.isfile(xml_file_path):
        pass
    else:
        pass
    with open(xml_file_path, 'wb') as f:
        f.write(xml_file)
    print(xml_file_path, 'written ')
    return True


def xml_2_pts(xml_file_path):
    if not os.path.isfile(xml_file_path):
        return None
    pts_xml = xml.etree.ElementTree.parse(xml_file_path)
    root = pts_xml.getroot()
    pts = []
    for i in range(len(root)):
        pt_dict = root[i].attrib
        pts.append([float(pt_dict['x']), float(pt_dict['y'])])
    return pts


def pts_2_xml(pts, xml_file_path):
    # if not pts or len(pts) == 0:
    #     if os.path.isfile(xml_file_path):
    #         os.remove(xml_file_path)
    #     return True
    if isinstance(pts, np.ndarray):
        pts = pts.tolist()
    vector_xml = xml.etree.ElementTree.Element('vector')
    for i in range(len(pts)):
        pt = pts[i]
        if len(pt) == 2:
            pt_tag = 'point_'+str(i)
            pt_xml = {'x': '{:.6f}'.format(pt[0]), 'y': '{:.6f}'.format(pt[1])}
            xml.etree.ElementTree.SubElement(vector_xml, pt_tag, attrib=pt_xml)
    xml_file = xml.etree.ElementTree.tostring(vector_xml)

    if os.path.isfile(xml_file_path):
        pass
    else:
        pass
    with open(xml_file_path, 'wb') as f:
        f.write(xml_file)
    return True


def test_xml_convertor():
    xml_file_path = r'D:/proj/hair_slam//data/vectors.xml'
    # xml_file_path = r'D:/Algorithm/Xml/Track/trackLeft.xml'
    pts = np.arange(0, 6)[:, None]
    roi = np.concatenate([pts, pts], axis=-1)
    # rois = [roi] * 3
    # print(rois[0].shape)
    rois = [np.array([])]
    rois_2_xml(rois, xml_file_path)
    pts = xml_2_rois(xml_file_path)

    # for pt in pts:
    #     print(pt)

    # pts = xml_2_pts(xml_file_path)
    # pts_2_xml(pts, '/home/cheng/Downloads/Vectors_Cheng.xml')
    # test_timer()


def main():
    test_xml_convertor()


if __name__ == '__main__':
    main()

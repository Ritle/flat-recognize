import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_project_opening_holes import audit_project_opening_holes, measure_topology_gaps


def fixture():
    return dict(levels=[dict(connectors=[dict(uuid='a',x=0,y=0),dict(uuid='b',x=200,y=0)],
        segments=[dict(uuid='s',start='a',end='b',height=270)],elements=[
        dict(uuid='d',type='doorItem',wall=dict(uuid='s',position=25,holeShape=[dict(x=-.4,y=0),dict(x=.4,y=0),dict(x=.4,y=2),dict(x=-.4,y=2)]),
             position=dict(z=0),materials=[dict(itemUuid='d')]),
        dict(uuid='w',type='windowItem',wall=dict(uuid='s',position=75,holeShape=[dict(x=-.2,y=0),dict(x=.2,y=0),dict(x=.2,y=.6),dict(x=-.2,y=.6)]),position=dict(z=100))])])


class OpeningHoleAuditTest(unittest.TestCase):
    def test_gap_coverage_uses_actual_holes_origin_and_scale(self):
        project = fixture()
        project['levels'][0]['segments'][0]['thickness'] = 12
        report = dict(adapter='physical_graph', pixel_to_world=dict(scale=2, origin=[10, 20]),
                      bindings={'doors': [dict(element_uuid='d', wall_uuid='s')],
                                'windows': [dict(element_uuid='w', wall_uuid='s')]},
                      topology_gaps_omitted=[dict(span=[[-15, -10], [105, -10]]),
                                             dict(span=[[5, 5], [25, 5]])])
        audit = audit_project_opening_holes(project)
        through_wall, outside = measure_topology_gaps(project, report, audit)
        for key, value in dict(length_px=120, solid_wall_px=40, opening_px=60, uncovered_px=20).items():
            self.assertAlmostEqual(through_wall[key], value)
        self.assertEqual(outside['uncovered_px'], outside['length_px'])
        report['bindings']['windows'][0]['element_uuid'] = 'old-window'
        with self.assertRaisesRegex(ValueError, 'do not match'):
            measure_topology_gaps(project, report, audit)

    def test_valid_holes_leave_source_unchanged(self):
        p=fixture();before=copy.deepcopy(p);r=audit_project_opening_holes(p)
        self.assertEqual(r['status'],'pass');self.assertEqual(len(r['openings']),2)
        self.assertEqual(r['openings'][0]['hole_start_cm'],10)
        self.assertEqual(p,before)

    def test_invalid_binding_cut_height_overlap_and_material(self):
        for mode in ('wall','position','shape','wall_length','height','overlap','material'):
            p=fixture();d,w=p['levels'][0]['elements']
            if mode=='wall':d['wall']['uuid']='missing'
            elif mode=='position':d['wall']['position']=101
            elif mode=='shape':d['wall']['holeShape']=[]
            elif mode=='wall_length':d['wall']['position']=0
            elif mode=='height':w['position']['z']=250
            elif mode=='overlap':w['wall']['position']=25
            else:d['materials'][0]['itemUuid']='other'
            with self.subTest(mode=mode):self.assertEqual(audit_project_opening_holes(p)['status'],'invalid')

    def test_separate_vertical_openings_do_not_overlap(self):
        p=fixture();d,w=p['levels'][0]['elements'];d['wall']['position']=w['wall']['position']=50
        d['wall']['holeShape'][2]['y']=d['wall']['holeShape'][3]['y']=.5
        self.assertEqual(audit_project_opening_holes(p)['status'],'pass')


if __name__=='__main__':unittest.main()

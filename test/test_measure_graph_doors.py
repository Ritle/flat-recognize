import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from measure_graph_doors import measure_graph_doors
from test.test_reclassify_graph_doors import fixture as raster_fixture


def fixture():
    graph, source, rgb = raster_fixture()
    points = [(20,80),(90,80.1),(110,79.9),(210,80),(210,150),(20,150)]
    graph.update(scale_cm_per_px=2,
                 nodes=[dict(id=i,point=list(p),adjacency=[]) for i,p in enumerate(points)],
                 edges=[dict(start=i,end=(i+1)%6,thickness_px=8) for i in range(6)])
    graph['diagnostics']['topology_gaps'] = []
    bridge = graph['diagnostics']['opening_bridges'][0]
    bridge.update(kind='doors',span=[[84,80],[116,80]],type_review=dict(
        source='raster_leaf_reclassification',evidence=dict(leaf_count=1,exported_aperture_unchanged=True)))
    return graph,source,rgb


class DoorMeasurementTest(unittest.TestCase):
    def test_expands_on_existing_path_and_preserves_source(self):
        graph,source,rgb=fixture();before=copy.deepcopy(graph)
        result,report=measure_graph_doors(graph,source,rgb)
        self.assertTrue(report[0]['accepted'])
        self.assertGreater(report[0]['measured_width_px'],report[0]['old_width_px'])
        self.assertLessEqual(report[0]['carrier']['max_deviation_px'],.5)
        self.assertEqual(len(result['faces']),1)
        bridge=result['diagnostics']['opening_bridges'][0]
        self.assertEqual(bridge['aperture_measurement']['previous_span'],[[84,80],[116,80]])
        self.assertFalse(bridge['type_review']['evidence']['exported_aperture_unchanged'])
        self.assertEqual(graph,before)
        self.assertEqual(measure_graph_doors(result,source,rgb)[0],result)

    def test_unsafe_paths_and_regions_do_not_change_graph(self):
        for mode in ('junction','deviation','missing_path','excluded','overlap'):
            with self.subTest(mode=mode):
                graph,source,rgb=fixture()
                if mode=='junction':
                    graph['nodes'].append(dict(id=6,point=[90,60],adjacency=[]))
                    graph['edges'].append(dict(start=1,end=6,thickness_px=8))
                elif mode=='deviation':graph['nodes'][1]['point'][1]=83
                elif mode=='missing_path':graph['edges'].pop(1)
                elif mode=='excluded':graph['diagnostics']['excluded_regions']=[[[75,75],[125,75],[125,85],[75,85]]]
                else:graph['diagnostics']['opening_bridges'].append(dict(kind='windows',span=[[117,80],[125,80]]))
                self.assertEqual(measure_graph_doors(graph,source,rgb)[0],graph)

    def test_current_raster_and_unique_source_are_required(self):
        for mode in ('white','duplicate','not_model_door','no_review','already_wide'):
            with self.subTest(mode=mode):
                graph,source,rgb=fixture()
                if mode=='white':rgb[:]=255
                elif mode=='duplicate':source['windows']*=2
                elif mode=='not_model_door':source['windows'][0]['source_class']='window'
                elif mode=='no_review':graph['diagnostics']['opening_bridges'][0].pop('type_review')
                else:graph['diagnostics']['opening_bridges'][0]['span']=[[80,80],[120,80]]
                self.assertEqual(measure_graph_doors(graph,source,rgb)[0],graph)

    def test_source_metadata_checked(self):
        for mode in ('dimensions','coordinates'):
            graph,source,rgb=fixture()
            if mode=='dimensions':source['meta']['width']=1
            else:graph['coordinate_space']='model'
            with self.subTest(mode=mode),self.assertRaises(ValueError):measure_graph_doors(graph,source,rgb)


if __name__=='__main__':unittest.main()

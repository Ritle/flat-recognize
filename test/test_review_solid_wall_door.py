import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from review_solid_wall_door import apply_reviewed_partition
from test.test_review_graph_doors import fixture as base_fixture


def fixture():
    graph,source,rgb,_=base_fixture()
    graph['edges']=graph['edges'][:6]
    graph['nodes']=graph['nodes'][:6]
    graph['nodes'][5]['adjacency'].remove(6)
    graph['nodes'][2]['adjacency'].remove(7)
    graph['diagnostics']['topology_gaps']=[]
    review=dict(version=1,image_sha256='a'*64,wall_choice='solid',user_statement='solid wall',
                visual_basis='Door leaf visible in raster',partition_span=[[10,60],[110,60]],
                door_span=[[50,60],[70,60]],thickness_px=6)
    return graph,source,rgb,review


class ReviewedSolidPartitionTest(unittest.TestCase):
    def apply(self,g,s,r,d):return apply_reviewed_partition(g,s,r,d,'a'*64)

    def test_solid_partition_adds_door_and_two_areas_without_mutation(self):
        graph,source,rgb,review=fixture();before=copy.deepcopy(graph)
        result=self.apply(graph,source,rgb,review)
        self.assertEqual(graph,before)
        self.assertEqual(len(result['edges']),len(graph['edges'])+1)
        self.assertEqual(len(result['faces']),2)
        bridge=result['diagnostics']['opening_bridges'][0]
        self.assertEqual(bridge['kind'],'doors')
        self.assertEqual(bridge['review']['wall_choice'],'solid')
        self.assertEqual(bridge['review']['door_reviewer'],'assistant_visual_review')
        self.assertEqual(bridge['clear_span'],[(50.0,60.0),(70.0,60.0)])
        self.assertEqual(self.apply(result,source,rgb,review),result)

    def test_partition_can_split_existing_wall_edges(self):
        graph,source,rgb,review=fixture()
        review['partition_span']=[[10,40],[110,40]]
        review['door_span']=[[50,40],[70,40]]
        result=self.apply(graph,source,rgb,review)
        self.assertEqual(len(result['nodes']),len(graph['nodes'])+2)
        self.assertEqual(len(result['faces']),2)

    def test_invalid_or_unsafe_partition_is_rejected(self):
        for mode in ('hash','choice','basis','missing_endpoint','overlap','excluded','crossing','gap','outside_door','jamb','unbounded','dimensions'):
            graph,source,rgb,review=fixture()
            if mode=='hash':review['image_sha256']='b'*64
            elif mode=='choice':review['wall_choice']='open'
            elif mode=='basis':review.pop('visual_basis')
            elif mode=='missing_endpoint':review['partition_span'][0]=[15,60]
            elif mode=='overlap':graph['diagnostics']['opening_bridges'].append(dict(kind='doors',span=[[55,60],[65,60]]))
            elif mode=='excluded':graph['diagnostics']['excluded_regions']=[[[55,50],[65,50],[65,70],[55,70]]]
            elif mode=='crossing':
                graph['nodes'].extend([dict(id=6,point=[60,50],adjacency=[]),dict(id=7,point=[60,70],adjacency=[])])
                graph['edges'].append(dict(start=6,end=7,thickness_px=6))
            elif mode=='gap':graph['diagnostics']['topology_gaps']=[dict(span=[[60,50],[60,70]])]
            elif mode=='outside_door':review['door_span']=[[50,61],[70,61]]
            elif mode=='jamb':review['door_span']=[[11,60],[30,60]]
            elif mode=='unbounded':graph['edges'].pop(0)
            else:source['meta']['width']=90
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.apply(graph,source,rgb,review)


if __name__=='__main__':unittest.main()

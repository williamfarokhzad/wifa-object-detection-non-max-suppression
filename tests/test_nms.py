import itertools
import random
import unittest
from pathlib import Path

from nms import Detection, iou, load, suppress

ROOT = Path(__file__).resolve().parents[1]


class NmsTests(unittest.TestCase):
    def test_raster_iou_reference(self):
        rng=random.Random(324)
        for _ in range(150):
            def rect(ident):
                x,y=rng.randrange(-5,7),rng.randrange(-5,7)
                return Detection(ident,'part',.5,[x,y,x+rng.randrange(1,9),y+rng.randrange(1,9)])
            a,b=rect('a'),rect('b')
            def pixels(d):
                x,y,u,v=map(int,d.box)
                return set(itertools.product(range(x,u),range(y,v)))
            ap,bp=pixels(a),pixels(b)
            self.assertAlmostEqual(iou(a,b),len(ap&bp)/len(ap|bp),places=13)

    def test_class_grouping_and_minimum_score(self):
        x=load(ROOT/'examples/overlap.json')
        self.assertEqual(suppress(x,.5,.5)['kept'],['A','C'])
        self.assertEqual(suppress(x,.5,.5,agnostic=True)['kept'],['A'])
        self.assertEqual(suppress(x,.5,.5)['trace'][-1]['decision'],'below-score')

    def test_suppressed_box_cannot_suppress(self):
        x=load(ROOT/'examples/chain.json')
        self.assertEqual(suppress(x,.5)['kept'],['A','C'])
        self.assertEqual(suppress(x,.5)['trace'][1]['by'],'A')
        self.assertGreater(iou(x[1],x[2]),.5)

    def test_threshold_equality_and_touching(self):
        x=load(ROOT/'examples/boundary.json')
        self.assertEqual(iou(x[0],x[1]),.5)
        self.assertEqual(iou(x[0],x[2]),0)
        self.assertEqual(suppress(x,.5,.5)['kept'],['A','B','C'])
        self.assertEqual(suppress(x,.49,.5)['kept'],['A','C'])
        self.assertEqual(suppress([x[1],x[0],x[2]],.49,.5)['kept'],['B','C'])

    def test_bad_input_is_rejected(self):
        for b in ([0,0,0,1],[0,0,1,float('nan')],[0,0,True,1],[0,0,1]):
            with self.subTest(box=b), self.assertRaises(ValueError):
                Detection('a','part',.5,b)
        x=Detection('a','part',.5,[0,0,1,1])
        for inp,thr in [([x,x],.5),([x],-1),([x],float('inf'))]:
            with self.subTest(threshold=thr), self.assertRaises(ValueError):
                suppress(inp,thr)

    def test_single_kept_with_duplicate_identical_boxes(self):
        a=Detection('a','part',.9,[0,0,4,4])
        b=Detection('b','part',.8,[0,0,4,4])
        self.assertEqual(iou(a,b),1)
        self.assertEqual(suppress([b,a])['kept'],['a'])
        self.assertEqual(suppress([b,a],1)['kept'],['a','b'])

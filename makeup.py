import numpy as np
import traceback
import cv2


lookup = np.zeros((256,256,256))

def minit():
    for r in range(256):
        for g in range(256):
            for b in range(256):
                lookup[r][g][b] = 0.299*r+0.587*g+.114*b


class mu(object):

    def __init__(self):
        self.ker = (-2, -1, 0, 1, 2)
        self.ct = 25
        self.z = np.int32([0, 0])
        self.buf = np.array([0.0, 0.0, 0.0])
        pass

    def add_lip(self, frame, gr, pts, color):
        miny, minx = np.min(pts, axis=0)
        maxy, maxx = np.max(pts, axis=0)

        try:
            roi = np.multiply(np.arange(maxx-minx+1)[:,np.newaxis] + np.arange(maxy-miny+1), 0)
            cv2.fillPoly(roi, [np.subtract(pts, [miny, minx])], 1)
            rat = np.minimum(np.ones((maxx-minx+1, maxy-miny+1)), np.divide(gr[minx:maxx+1, miny:maxy+1], 170.0))
            rat = np.multiply(roi, rat)
            mb = np.multiply(rat, color[0])
            mg = np.multiply(rat, color[1])
            mr = np.multiply(rat, color[2])
            rat = np.dstack([mb, mg, mr])
            roi = np.bitwise_xor(roi, 1)
            np.add(roi, 1, out=roi)
            zs = np.multiply(frame[minx:maxx+1, miny:maxy+1], np.dstack((roi, roi, roi)))
            frame[minx:maxx+1, miny:maxy+1] = np.int32(np.divide(np.add(zs, rat), 2))
        except:
            traceback.print_exc()
        return

        for x in range(minx, maxx):
            for y in range(miny, maxy):
                ra = cv2.pointPolygonTest(pts, (y, x), False)
                if ra>=0:
                    try:
                        rat = min(1, gr[x, y]/170)
                        np.add(np.multiply(color, rat), frame[x, y], out=self.buf)
                        frame[x, y] = np.int32(np.divide(self.buf, 2))

                    except:
                        #traceback.print_exc()
                        pass

    def add_eyeshadow(self, frame, gr, pts, md, color):
        vec = []
        for pt in pts:
            vec.append([pt[0]-md[0], (pt[1]-md[1])])
        mdd = 0
        for dx, dy in vec:
            mdd = max(mdd, np.linalg.norm([dx, dy]-self.z))
        k = []
        for v, p in zip(vec, pts):
            k.append([v[0]+p[0], v[1]+p[1]])
        mdd*=2
        pts = np.array(np.concatenate((pts, k[::-1])), dtype=np.int32)
        miny, minx = np.min(pts, axis=0)
        maxy, maxx = np.max(pts, axis=0)

        try:
            roi = np.multiply(np.arange(maxx-minx+1)[:,np.newaxis] + np.arange(maxy-miny+1), 0)
            cv2.fillPoly(roi, [np.subtract(pts, [miny, minx])], 1)
            sx = md[1]-minx
            sy = md[0]-miny
            dis = np.fromfunction(lambda x, y: np.sqrt(np.multiply(x-sx, x-sx)+np.multiply(y-sy, y-sy)), (maxx-minx+1, maxy-miny+1))
            raq = np.minimum(np.ones((maxx-minx+1, maxy-miny+1)), np.sqrt(np.divide(dis, mdd)))
            rat = np.minimum(np.ones((maxx-minx+1, maxy-miny+1)), np.divide(gr[minx:maxx+1, miny:maxy+1], 170.0))
            np.multiply(rat, np.subtract(1, raq), out=rat)
            np.multiply(rat, roi, out=rat)
            mb = np.multiply(rat, color[0])
            mg = np.multiply(rat, color[1])
            mr = np.multiply(rat, color[2])
            rat = np.dstack([mb, mg, mr])
            roi = np.bitwise_xor(roi, 1)
            np.maximum(roi, raq, out=raq)
            raq = np.dstack((raq, raq, raq))
            frame[minx:maxx+1, miny:maxy+1] = np.int32(np.add(rat, np.multiply(frame[minx:maxx+1, miny:maxy+1], raq)))
        except:
            traceback.print_exc()
        return

        for x in range(minx, maxx):
            for y in range(miny, maxy):
                ra = cv2.pointPolygonTest(pts, (y, x), False)
                if ra>=0:
                    d = np.linalg.norm([y, x] - md)
                    try:
                        raq = min(1, np.sqrt(d/mdd))
                        rat = min(1, gr[x, y]/170)*(1-raq)
                        np.add(np.multiply(color, rat), np.multiply(frame[x, y], raq), out=self.buf)
                        frame[x, y] = np.int32(self.buf)
                    except:
                        pass

    def add_blush(self, frame, gr, md, th, color):
        maxx = md[1] + th
        minx = md[1] - th
        maxy = md[0] + th
        miny = md[0] - th
        dth = np.linalg.norm(th-self.z)*0.8

        try:
            sx = (maxx-minx)/2
            sy = (maxy-miny)/2
            roi = np.multiply(np.arange(maxx-minx+1)[:,np.newaxis] + np.arange(maxy-miny+1), 0)
            cv2.circle(roi, (md[1]-minx, md[0]-miny), th, 1, -1)
            dis = np.fromfunction(lambda x, y: np.sqrt(np.multiply(x-sx, x-sx)+np.multiply(y-sy, y-sy)), (maxx-minx+1, maxy-miny+1))
            np.divide(dis, dth, out=dis)
            np.subtract(1, dis, out=dis)
            np.divide(dis, 3, out=dis)
            np.multiply(dis, roi, out=dis)
            rat = np.minimum(np.ones((maxx-minx+1, maxy-miny+1)), np.divide(gr[minx:maxx+1, miny:maxy+1], 170.0))
            np.multiply(rat, dis, out=rat)
            raq = np.subtract(1, rat)
            mb = np.multiply(rat, color[0])
            mg = np.multiply(rat, color[1])
            mr = np.multiply(rat, color[2])
            rat = np.dstack([mb, mg, mr])
            frame[minx:maxx+1, miny:maxy+1] = np.int32(np.add(rat,
                np.multiply(frame[minx:maxx+1, miny:maxy+1], np.dstack((raq, raq, raq))),))
        except:
            #traceback.print_exc()
            pass
        return

        for x in range(minx, maxx):
            for y in range(miny, maxy):
                d = np.linalg.norm((y, x) - md)
                if d <= dth:
                    try:
                        rat = min(1, gr[x, y]/170)*(1-d/dth)/3
                        raq = 1-rat
                        np.add(np.multiply(color, rat), np.multiply(frame[x, y], raq), out=self.buf)
                        frame[x, y] = np.int32(self.buf)
                    except:
                        pass

        #cv2.addWeighted(mask, 0.3, frame, 0.7, 0, frame)

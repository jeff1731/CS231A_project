import numpy as np
import cv2
import pdb
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage.filters import gaussian_filter


def find_crash(crash_num,show_figs=True):
    file_name ="../videos/crash" + str(crash_num) + ".mp4"


    # cap = cv2.VideoCapture('videos/crash1.mp4')
    cap = cv2.VideoCapture(file_name)



    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (20,20),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    pss = []
    pos = {}
    for i,p in enumerate(p0):
        if i not in pos:
            pos[i] = [p]
        else:
            pos[i].append(p)



    while(1):
        ret,frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)


        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        
        num_removed = 0
        for i,p in enumerate(p1):
            if st[i][0] == 0:
                #pdb.set_trace()
                del pos[i-num_removed]
                num_removed += 1
                j = i-num_removed+1
                while True:
                    if j+1 not in pos:
                        break
                    pos[j] = pos[j+1]
                    del pos[j+1]
                    j += 1
            else:
                pos[i-num_removed].append(p)

        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        img = cv2.add(frame,mask)

        if show_figs:
            cv2.imshow('frame',img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)

    cv2.destroyAllWindows()
    cap.release()

    # print len(pos)
    # print len(p1)



    t = [i for i in range(len(pos[0]))]
    x_all =[]
    y_all =[]
    v_all = []
    a_all = []
    dTheta_all = []
    c_all = []

    thresh = 10

    largest = 0
    frame_idx = 0
    feat_cent = []
    feat_idx  = 0

    for feat in range(len(pos)):
        x = []
        y = []
        for p in pos[feat]:
            xi,yi = p[0]
            x.append(xi)
            y.append(yi)
        #if (abs(max(x)-min(x)) > thresh) and (abs(max(y)-min(y)) > thresh):

        v = []
        dt = 0.5
        for i in range(1,len(pos[feat])):
            x0 = np.array(pos[feat][i-1])
            x1 = np.array(pos[feat][i])
            dist = np.linalg.norm(x0-x1)
            v.append(dist/dt)

        v = gaussian_filter(v, sigma=2)

        theta = []
        for i in range(2,len(pos[feat])):
            x0 = np.array(pos[feat][i-2])
            x1 = np.array(pos[feat][i-1])
            x2 = np.array(pos[feat][i])
            v0 = x1-x0
            v1 = x2-x1
            #print np.linalg.norm(v0), np.linalg.norm(v1)
            if np.linalg.norm(v0) < 1.0 or np.linalg.norm(v1) < 1.0:
                if len(theta) > 0:
                    angle = theta[-1]
                else:
                    angle = 0
            else:
                v0 = v0/np.linalg.norm(v0)
                v1 = v1/np.linalg.norm(v1)
                c = np.dot(v0[0],v1[0])
                angle = np.arccos(np.clip(c,-1,1))
                angle = angle * 180.0/np.pi
            
           
            theta.append(angle)
        dTheta = []
        for i in range(1,len(theta)):
            dT = abs(theta[i-1]-theta[i])
            if i < 10:
                dT = min(10,dT)
            dTheta.append(dT)

        dTheta = gaussian_filter(dTheta,sigma=3)

        a = []
        for i in range(1,len(v)):
            v0 = np.array(v[i-1])
            v1 = np.array(v[i])
            dv = np.linalg.norm(v0-v1)
            ac = dv/dt
            if i < 10:
                ac = min(ac,3.0)
           

            a.append(ac)
            # if ac > largest_accel:
            #     largest_accel = ac
            #     frame_idx = i+2
            #     feat_cent.append(pos[feat][i+2][0])
            #     feat_idx = feat
        
        c = np.multiply(a[1:],dTheta)
        for i in range(len(c)):
            ci = c[i]
            if ci > largest:
                largest = ci
                # frame_idx = int((i+2)*len(a)/len(c)) + 2
                frame_idx = i+2
                feat_cent.append(pos[feat][frame_idx][0])
                feat_idx = feat

        x_all.append(x)
        y_all.append(y)
        v_all.append(v)
        a_all.append(a)
        dTheta_all.append(dTheta)
        c_all.append(c)


    # for a in a_all:

    if show_figs:
        plt.figure()
        # plt.subplot(511)
        # plt.title("Position")
        # for i in range(len(x_all)):
        #     if i == feat_idx:
        #         plt.plot(t,x_all[i])
        #         plt.plot(t,y_all[i])
        # plt.grid("on")
        # plt.subplot(512)
        # plt.title("Velocity")
        # for i in range(len(v_all)):
        #     if i == feat_idx:
        #         plt.plot(t[1:],v_all[i])
        #     plt.grid("on")
        plt.subplot(311)
        plt.title("Acceleration")
        for i in range(len(a_all)):
            #if i == feat_idx:
                plt.plot(t[2:],a_all[i])
        plt.grid("on")
        plt.subplot(312)
        plt.title("Change in direction")
        for i in range(len(dTheta_all)):
            #if i == feat_idx:
                dTh = dTheta_all[i]
                plt.plot([j for j in range(len(dTh))],dTheta_all[i])
        plt.grid("on")
        plt.subplot(313)
        plt.title("Acceleration times change in direction")
        for i in range(len(c_all)):
            #if i == feat_idx:
                plt.plot([j for j in range(len(c_all[i]))],c_all[i])
        plt.grid("on")
        plt.show()
        plt.close()

    # Show crash frame with box around detected crash
    cap = cv2.VideoCapture(file_name)
    #print frame_idx
    cap.set(1,frame_idx);
    ret,frame = cap.read()


    mask = np.zeros_like(old_frame)
    width = 200


    feat_cent = np.mean(feat_cent[-2:],axis=0)
    a = feat_cent - width/2
    b = [a[0],a[1]+width]
    c = [a[0] +width, a[1]]
    d = feat_cent + width/2
    mask = cv2.line(mask, (int(a[0]),int(a[1])),(int(b[0]),int(b[1])), [0, 255, 0], 2)
    mask = cv2.line(mask, (int(a[0]),int(a[1])),(int(c[0]),int(c[1])), [0, 255, 0], 2)
    mask = cv2.line(mask, (int(d[0]),int(d[1])),(int(b[0]),int(b[1])), [0, 255, 0], 2)
    mask = cv2.line(mask, (int(d[0]),int(d[1])),(int(c[0]),int(c[1])), [0, 255, 0], 2)

    frame = cv2.circle(frame,(int(feat_cent[0]),int(feat_cent[1])),5,[0, 255, 0],-1)



    img = cv2.add(frame,mask)
    if show_figs:
        cv2.imshow('frame',img)
        while True:
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

    cv2.imwrite("crash_pics/crash"+str(crash_num)+'_frame_'+str(frame_idx)+'.jpg',img)

    cv2.destroyAllWindows()
    cap.release()

    return largest





from imports import *

"""## PFM read-write functions"""

def read_pfm(path):
    """Read pfm file.
    Args:
        path (str): path to file
    Returns:
        tuple: (data, scale)
    """
    with open(path, "rb") as file:
      color = None
      width = None
      height = None
      scale = None
      endian = None

      header = file.readline().rstrip()
      if header.decode("ascii") == "PF":
          color = True
      elif header.decode("ascii") == "Pf":
          color = False
      else:
          raise Exception("Not a PFM file: " + path)

      dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
      if dim_match:
          width, height = list(map(int, dim_match.groups()))
      else:
          raise Exception("Malformed PFM header.")

      scale = float(file.readline().decode("ascii").rstrip())
      if scale < 0:
          # little-endian
          endian = "<"
          scale = -scale
      else:
          # big-endian
          endian = ">"

      data = np.fromfile(file, endian + "f")
      shape = (height, width, 3) if color else (height, width)

      data = np.reshape(data, shape)
      data = np.flipud(data)

      return data, scale


def write_pfm(path, image, scale=1):
  """Write pfm file.
  Args:
      path (str): pathto file
      image (array): data
      scale (int, optional): Scale. Defaults to 1.
  """
  with open(path, "wb") as file:
      color = None

      if image.dtype.name != "float32":
          raise Exception("Image dtype must be float32.")

      image = np.flipud(image)

      if len(image.shape) == 3 and image.shape[2] == 3:  # color image
          color = True
      elif (
          len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1
      ):  # greyscale
          color = False
      else:
          raise Exception("Image must have H x W x 3, H x W x 1 or H x W dimensions.")

      file.write("PF\n" if color else "Pf\n".encode())
      file.write("%d %d\n".encode() % (image.shape[1], image.shape[0]))

      endian = image.dtype.byteorder

      if endian == "<" or endian == "=" and sys.byteorder == "little":
          scale = -scale

      file.write("%f\n".encode() % scale)

      image.tofile(file)


def calibrator(calib_img_dir, results_save_dir, cam_mat_save_path='cameraIntrinsic_apple.xml', chessboardSize = (24,17)):

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

    size_of_chessboard_squares_mm = 20
    objp = objp * size_of_chessboard_squares_mm


    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob(f'{calib_img_dir}/*')

    for image in images:

        img = cv2.imread(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        frameSize = (gray.shape[0], gray.shape[1])

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, chessboardSize, None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)

    ############## CALIBRATION #######################################################
    ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
    ############## UNDISTORTION #####################################################
    test_inp_path = images[0]
    img_name = test_inp_path.split('/')[-1]
    img = cv2.imread(test_inp_path)
    h,  w = img.shape[:2]
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))
    cm_save = cv2.FileStorage(cam_mat_save_path,cv2.FILE_STORAGE_WRITE)
    cm_save.write('intrinsic',newCameraMatrix)
    cm_save.release()

    # Undistort with Remapping
    mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite(os.path.join(results_save_dir,img_name), dst)

    # Reprojection Error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error

    print( "total error: {}".format(mean_error/len(objpoints)))
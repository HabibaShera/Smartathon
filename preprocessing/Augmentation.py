class Augment_Images:
  def __init__(self, img_path, angle_rotation):
    self.img_path = img_path
    self.img_id = img_path.split('.')[0]
    full_path = os.path.join('/content/FullData/dataset/images', img_path)
    self.img = cv2.imread(full_path)
    self.angle=angle_rotation

  def info(self, keyword):
    width, height = self.img.shape[:2]
    bboxes = abs(train[train['image_path']==self.img_path][['xmin', 'ymin', 'xmax', 'ymax']]*2).values
    labels = train[train['image_path']==self.img_path]['class'].values
    
    with open(f"{self.img_id}{keyword}.txt", 'w') as file:
      for i in range(len(bboxes)):
        x_cen = round((bboxes[i][0] + bboxes[i][2]) / (2*width), 3)
        y_cen = round((bboxes[i][1] + bboxes[i][3]) / (2*height), 3)
        shape_width = round((bboxes[i][2] + bboxes[i][0]) / (width), 3)
        shape_height = round((bboxes[i][3] + bboxes[i][1]) / (height), 3)
        file.write(f'{labels[i]} {x_cen} {y_cen} {shape_width} {shape_height}\n') 


  def contrastImage(self):
    gray = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
    self.image = cv2.equalizeHist(gray)
    cv2.imwrite(f'{self.img_id}contrast.jpg', self.image)
    self.info('contrast')
    
  def sharpImage(self):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    self.image = cv2.filter2D(self.img, -1, kernel)
    cv2.imwrite(f'{self.img_id}sharp.jpg', self.image)
    self.info('sharp')
    
  def blurImage(self):
    self.image = cv2.GaussianBlur(self.img,(15,15),0) 
    cv2.imwrite(f'{self.img_id}blur.jpg', self.image) 
    self.info('blur')  

  
  def selectAll(self):
    self.contrastImage()
    self.sharpImage()
    self.blurImage()



class AugmentedImagesWithBBox:
  def __init__(self, img_path):
    self.img_id = img_path.split('.')[0]
    full_path = os.path.join('/content/FullData/dataset/images', img_path)
    self.img = cv2.imread(full_path)
    
  def rotateImage(self):
    height, width = self.img.shape[:2]
    img_c = (width/2,height/2) # Image Center Coordinates
    rotation_matrix = cv2.getRotationMatrix2D(img_c, self.angle, 1.0)
    
    abs_cos = abs(rotation_matrix[0,0])  # Cos(angle)
    abs_sin = abs(rotation_matrix[0,1])  # sin(angle)

    # New Width and Height of Image after rotation
    bound_w = int(height*abs_sin + width*abs_cos)
    bound_h = int(height*abs_cos + width*abs_sin)
        
    # subtract the old image center and add the new center coordinates
    rotation_matrix[0,2]+=bound_w/2-img_c[0]
    rotation_matrix[1,2]+=bound_h/2-img_c[1]
        
    # rotating image with transformed matrix and new center coordinates
    rotated_matrix = cv2.warpAffine(self.img, rotation_matrix,(bound_w, bound_h))
        
    self.image = rotated_matrix
    cv2.imwrite(f'{self.img_id}rotate.jpg', self.image)



def pascal_voc_to_yolo(image_path):
  width, height = 1080, 1920
  bboxes = abs(train[train['image_path']==image_path][['xmin', 'ymin', 'xmax', 'ymax']]*2).values
  labels = train[train['image_path']==image_path]['class'].values
  
  with open(f"{image_path.split('.')[0]}.txt", 'w') as file:
    for i in range(len(bboxes)):
      x_cen = round((bboxes[i][0] + bboxes[i][2]) / (2*width), 3)
      y_cen = round((bboxes[i][1] + bboxes[i][3]) / (2*height), 3)
      shape_width = round((bboxes[i][2] + bboxes[i][0]) / (width), 3)
      shape_height = round((bboxes[i][3] + bboxes[i][1]) / (height), 3)
      file.write(f'{labels[i]} {x_cen} {y_cen} {shape_width} {shape_height}\n')
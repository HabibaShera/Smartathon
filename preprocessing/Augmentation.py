class Augment_Images:
  def __init__(self, img_path):
    self.img_path = img_path
    self.img_id = img_path.split('.')[0]
    full_path = os.path.join('/content/FullData/dataset/images', img_path)
    self.img = cv2.imread(full_path)

  def info(self, keyword):
    width, height = self.img.shape[:2]

    bb = abs(train[train['image_path']==self.img_path][['xmin', 'ymin', 'xmax', 'ymax']]*2).values
    bboxes = np.where(bb<0, 0, bb)
    labels = train[train['image_path']==self.img_path]['class'].values
    
    with open(f"{keyword}.txt", 'w') as file:
      for i in range(len(bboxes)):
        x_cen = min(round((bboxes[i][0] + bboxes[i][2]) / (2*width), 3), 1)
        y_cen = min(round((bboxes[i][1] + bboxes[i][3]) / (2*height), 3), 1)
        shape_width = min(round((bboxes[i][2] + bboxes[i][0]) / (width), 3), 1)
        shape_height = min(round((bboxes[i][3] + bboxes[i][1]) / (height), 3), 1)
        file.write(f'{labels[i]} {x_cen} {y_cen} {shape_width} {shape_height}\n') 


  def equalizeImage(self): #return 1 image
    gray = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
    self.image = cv2.equalizeHist(gray)
    cv2.imwrite(f'{self.img_id}equalize.jpg', self.image)
    self.info(f'{self.img_id}equalize')
    
  def sharpImage(self): #return 2 images
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    self.image = cv2.filter2D(self.img, -1, kernel)
    cv2.imwrite(f'{self.img_id}sharp1.jpg', self.image)
    self.info(f'{self.img_id}sharp1')

    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    self.image = cv2.filter2D(self.img, -1, kernel)
    cv2.imwrite(f'{self.img_id}sharp2.jpg', self.image)
    self.info(f'{self.img_id}sharp2')

    
  def blurImage(self, kernel_size=(15,15)): #return 1 image
    self.image = cv2.GaussianBlur(self.img,kernel_size, 0) 
    cv2.imwrite(f'{self.img_id}blur.jpg', self.image) 
    self.info(f'{self.img_id}blur')  

  def randomBrightContrast(self): #return 9 images
    values = np.arange(1,10)/10
    for i in values:
      transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=i, contrast_limit=i, p=0.8)
                            ])
      transformed = transform(image=self.img)
      cv2.imwrite(f'{self.img_id}{str(i)}randomBright.jpg', transformed['image'])
      self.info(f'{self.img_id}{str(i)}randomBright')

  def randomShadow(self): #return 5 images
    for i in range(1, 6):
      transform = A.Compose([
          A.RandomShadow(num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=3, shadow_roi=(0, 0.5, 1, 1), p=1)
                            ])
      transformed = transform(image=self.img)
      cv2.imwrite(f'{self.img_id}{str(i)}randomshadow.jpg', transformed['image'])
      self.info(f'{self.img_id}{str(i)}randomshadow')

  def groupedProcessing(self): #return 6 images
    medium = A.Compose([
        A.CLAHE(p=1),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50, p=1),
            ], p=1)
    transformed2 = medium(image=self.img)

    strong = A.Compose([
        A.ChannelShuffle(p=1),
                ], p=1)
    transformed3 = strong(image=self.img)

    cv2.imwrite(f'{self.img_id}group2.jpg', transformed2['image'])
    cv2.imwrite(f'{self.img_id}group3.jpg', transformed3['image'])
      
    self.info(f'{self.img_id}group2')
    self.info(f'{self.img_id}group3')

    for i in range(1, 5):
      light = A.Compose([
      A.RandomBrightnessContrast(p=1),    
      A.RandomGamma(p=1),    
      A.CLAHE(p=1),    
            ], p=1)
      transformed1 = light(image=self.img)

      
      cv2.imwrite(f'{self.img_id}{str(i)}group1.jpg', transformed1['image'])
      
      self.info(f'{self.img_id}{str(i)}group1')
      

  def selectAll(self): #return 24 images
    self.equalizeImage()
    self.sharpImage()
    self.blurImage()
    self.randomShadow()
    self.randomBrightContrast()
    self.groupedProcessing()


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
import json
import os
import shutil
# number of classes
nc = 5

# class names
names = ['background','car','wheel','lights','window']
class json2txt():
    def __init__(self, file_name):
        self.file_name = file_name
        self.json_file = None
        self.label_dir = './labels'
        self.task()
        
        
    def create_folder(self):
        if not os.path.exists(self.label_dir):
            os.makedirs(self.label_dir)
        else:
            shutil.rmtree(self.label_dir)
            print(f"delte {self.label_dir} and remake")
            os.makedirs(self.label_dir)



    
    def print_file_name(self):
        print(self.file_name)
    
    def load_json(self):
        with open(self.file_name, 'r') as f:
            self.json_file = json.load(f)
    

    def read_all_file(self):
        for i in self.json_file["assets"]:
            self.read_task(i)
    

            
    
    def read_task(self, i):
        
        name = self.json_file["assets"][i]['asset']["name"]
        name = os.path.splitext(name)[0]

        txt_name = name + '.txt'
        txt_dir = os.path.join(self.label_dir, txt_name)
        with open(txt_dir, 'w') as f:
            width = self.json_file["assets"][i]["asset"]['size']['width']
            height = self.json_file["assets"][i]["asset"]['size']['height']

            bbox_num = len(self.json_file["assets"][i]["regions"])
            print(name, width, height)
            # 왼쪽 아래 넓이 높이 --> x 중심, y 중심 넓이 높이
            for idx, j in enumerate(range(bbox_num)):
                cls_name = self.json_file["assets"][i]["regions"][j]['tags'][0]
                cls_index = names.index(cls_name)
                bbox_c = self.json_file["assets"][i]["regions"][j]['boundingBox']
                # print(cls_name, cls_index, bbox_c)
                b_height = bbox_c["height"]/height
                b_width = bbox_c["width"]/width
                b_Xcenter = (bbox_c["left"] + (bbox_c["width"]/2) )/width
                b_Ycenter = (bbox_c["top"] + (bbox_c["height"]/2) )/height
                print(cls_name, cls_index, b_Xcenter, b_Ycenter, b_height, b_width)
                _str = str(cls_index) + ' ' + str(b_Xcenter) + ' ' + str(b_Ycenter) + ' ' + str(b_height) + ' ' + str(b_width)
                f.write(_str)
                if idx < bbox_num-1:
                    f.write('\n')
            print("\n\n")

            

        

        # print(f"name : {name}, \nwidth :  {width}, \nheight : {height}  \nbbox_num = {bbox_num}\n\n")


        
    
    def task(self):
        self.create_folder()
        self.print_file_name()
        self.load_json()
        self.read_all_file()






if __name__ == "__main__":
    a = json2txt('test.json')
    print(len(names))

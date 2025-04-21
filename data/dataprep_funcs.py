import os
import torch

class label():
    def __init__(self, obj_class, norm_coords) -> None:
        self.obj_class=obj_class
        self.norm_coords=norm_coords
        
class DotaDataset_1p5():
    def __init__(self, root, folder):
        self.root=root
        self.imgs=[os.path.join(root, "images", folder, x) for x in list(sorted(os.listdir(os.path.join(root, "images", folder))))]
        self.labels=[os.path.join(root, "labels", folder, x) for x in list(sorted(os.listdir(os.path.join(root,"labels", folder))))]
        self.label_content={}
        for x in self.labels:
            with open(x, 'r') as file:
                self.label_content[x] = []
                for line in file:
                    line_data=line.replace(' ', ',').split(',')
                    coords=[float(y) for y in line_data[0:]]
                    self.label_content[x].append(
                        label(
                            obj_class=int(line_data[0]), 
                            norm_coords=coords
                        )
                    )

    def instances_per_class(self):
        class_count = {}
        for _, val in self.label_content.items():
            for x in val:
                if x.obj_class not in class_count:
                    class_count[x.obj_class]=0
                class_count[x.obj_class]+=1
        return class_count

    def files_per_class(self):
        num_of_files_per_class = {}
        for _, val in self.label_content.items():
            classes_in_file = set([x.obj_class for x in val])           
            for v in classes_in_file:
                if v not in num_of_files_per_class:
                    num_of_files_per_class[v] = 0
                num_of_files_per_class[v]+=1 
        return  num_of_files_per_class
    

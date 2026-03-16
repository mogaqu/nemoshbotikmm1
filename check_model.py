from ultralytics import YOLO

m = YOLO("MM2_Bot_Package/weights/balls+coins.pt")
with open("output.txt", "w") as f:
    f.write("Classes: " + str(m.names) + "\n")
    f.write("Number of classes: " + str(len(m.names)) + "\n")
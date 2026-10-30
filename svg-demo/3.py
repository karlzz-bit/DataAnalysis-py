import xml.etree.ElementTree as ET

# 创建SVG根元素
svg = ET.Element('svg', xmlns="http://www.w3.org/2000/svg", width="400", height="200")

# 创建圆形元素
circle = ET.Element('circle', cx="100", cy="60", r="50", fill="yellow", stroke="black")

# 将圆形元素添加到svg
svg.append(circle)

# 将SVG结构写入文件
tree = ET.ElementTree(svg)
with open("circle_shape.svg", "wb") as f:
    tree.write(f)

print("SVG file created: circle_shape.svg")

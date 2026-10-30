import xml.etree.ElementTree as ET

# 创建SVG根元素
svg = ET.Element('svg', xmlns="http://www.w3.org/2000/svg", width="400", height="400")

# 创建text元素
text = ET.Element('text', x="100", y="100", fill="black", **{'font-size': "30"})
text.text = "Hello"

# 将text元素添加到svg
svg.append(text)

# 将SVG结构写入文件
tree = ET.ElementTree(svg)
with open("hello_text.svg", "wb") as f:
    tree.write(f)

print("SVG file created: hello_text.svg")

# import matplotlib.pyplot as plt
# import networkx as nx
# #
# # # 创建一个有向图
# G = nx.DiGraph()
#
# # 添加节点和边
# nodes = ['blog', '_config.butterfly.yml', '_config.yml', 'db.json', 'node_modules/', 'package-lock.json', 'package.json', 'scaffolds/', 'source/', 'themes/']
# edges = [('blog', '_config.butterfly.yml'), ('blog', '_config.yml'), ('blog', 'db.json'), ('blog', 'node_modules/'), ('blog', 'package-lock.json'), ('blog', 'package.json'), ('blog', 'scaffolds/'), ('blog', 'source/'), ('blog', 'themes/')]
#
# G.add_nodes_from(nodes)
# G.add_edges_from(edges)
#
# # 画图
# pos = nx.spring_layout(G)
# nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight="bold", arrows=True)
# plt.title('Project Structure')
# plt.show()
from graphviz import Digraph
import os
os.environ["PATH"] += os.pathsep + "D:/Graphviz-11.0.0-win64/bin"
# dot = Digraph(comment='Project Structure')
# # 添加节点
# dot.node('A', 'blog')
# dot.node('B', '_config.butterfly.yml')
# dot.node('C', '_config.yml')
# dot.node('D', 'db.json')
# dot.node('E', 'node_modules/')
# dot.node('F', 'package-lock.json')
# dot.node('G', 'package.json')
# dot.node('H', 'scaffolds/')
# dot.node('I', 'source/')
# dot.node('J', 'themes/')
# 
# # 添加边
# dot.edge('A', 'B')
# dot.edge('A', 'C')
# dot.edge('A', 'D')
# dot.edge('A', 'E')
# dot.edge('A', 'F')
# dot.edge('A', 'G')
# dot.edge('A', 'H')
# dot.edge('A', 'I')
# dot.edge('A', 'J')
#
# # 保存并展示图表
# dot.render('project_structure', format='png', view=True)



from graphviz import Digraph

# 创建一个新的有向图，并设置编码为UTF-8
dot = Digraph(comment='Hexo + Butterfly Theme Use Case Diagram', encoding='utf-8')

# 添加系统边界
with dot.subgraph(name='cluster_system') as c:
    c.attr(label='Hexo + Butterfly Theme Blog System')
    c.node_attr.update(fontname='SimSun')  # 使用支持中文的字体，如SimSun（宋体）

    # 添加用例
    c.node('UC1', '导航栏设置')
    c.node('UC2', '菜单设置')
    c.node('UC3', '文章归档')
    c.node('UC4', '标签和分类')
    c.node('UC5', '文章列表')
    c.node('UC6', '留言板')
    c.node('UC7', '友情链接')
    c.node('UC8', '关于页面')
    c.node('UC9', '自定义页面')
    c.node('UC10', '文章格式支持')
    c.node('UC11', '响应式设计')
    c.node('UC12', 'SEO优化')
    c.node('UC13', '插件和主题扩展')

# 添加参与者（用户）
dot.node('User', '用户', shape='actor', fontname='SimSun')

# 创建参与者和用例之间的关系
edges = [
    ('User', 'UC1'), ('User', 'UC2'), ('User', 'UC3'), ('User', 'UC4'), ('User', 'UC5'),
    ('User', 'UC6'), ('User', 'UC7'), ('User', 'UC8'), ('User', 'UC9'), ('User', 'UC10'),
    ('User', 'UC11'), ('User', 'UC12'), ('User', 'UC13')
]

dot.edges(edges)

# 显示图
dot.render('hexo_butterfly_usecase', format='png', view=True)



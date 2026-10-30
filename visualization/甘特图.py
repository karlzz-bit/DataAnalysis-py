import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib.font_manager as fm

# 设置中文字体，SimHei 是 Windows 中的中文字体，如果是其他系统，请替换为相应的字体
plt.rcParams['font.family'] = 'SimHei'

# 定义阶段和时间
phases = [
    {'phase': '需求分析阶段', 'start_date': '2024-06-10', 'end_date': '2024-06-12'},
    {'phase': '设计阶段', 'start_date': '2024-06-13', 'end_date': '2024-06-15'},
    {'phase': '编码实现', 'start_date': '2024-06-16', 'end_date': '2024-06-22'},
    {'phase': '测试', 'start_date': '2024-06-23', 'end_date': '2024-06-26'},
    {'phase': '部署', 'start_date': '2024-06-27', 'end_date': '2024-06-28'},
    {'phase': '汇总', 'start_date': '2024-06-29', 'end_date': '2024-06-30'}
]

# 解析日期并转换为 datetime 对象
for phase in phases:
    phase['start_date'] = datetime.strptime(phase['start_date'], '%Y-%m-%d')
    phase['end_date'] = datetime.strptime(phase['end_date'], '%Y-%m-%d')

# 提取阶段名称和时间范围
phases_names = [phase['phase'] for phase in phases]
start_dates = [phase['start_date'] for phase in phases]
end_dates = [phase['end_date'] for phase in phases]
duration = [(end - start).days for start, end in zip(start_dates, end_dates)]

# 绘制甘特图
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制任务条
for i, phase in enumerate(phases):
    ax.barh(phase['phase'], duration[i], left=start_dates[i], height=0.5, align='center', alpha=0.8)

# 设置日期格式
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# 设置标题和标签
ax.set_title('项目阶段甘特图')
ax.set_xlabel('日期')
ax.set_ylabel('阶段')

# 设置 x 轴范围
ax.set_xlim(min(start_dates), max(end_dates))

# 显示图例
ax.legend(['阶段持续时间'], loc='best')

# 自动调整布局
plt.tight_layout()

# 显示图表
plt.show()

#!/bin/bash
# 该脚本用于统计指定目录下每个子目录中的图片文件数量
# 运行脚本命令:
# bash cal.sh /path/to/your/directory
# bash cal.sh /app/dataset_huazhi/水产数据/原始数据/水上/
# bash cal.sh /gpfs/users/aiAnalyze/数据总纲/水产/原始表型数据/青岛鱼数据/已标注数据
# bash cal.sh /gpfs/users/aiAnalyze/数据总纲/水产/原始表型数据/青岛鱼20241129
# bash cal.sh /gpfs/users/aiAnalyze/数据总纲/水产/原始表型数据/青岛鱼20241129/11.25青岛桌面/11.25青岛鱼数据
# bash cal.sh /gpfs/users/aiAnalyze/数据总纲/水产/已标注/caoyu0-15-水下

# 鱼*color.png

# 检查参数数量
numOfArgs=$#
if [ $numOfArgs -ne 1 ]; then
    echo -e "Usage: \nbash $0 dirForCount"
    exit -1
fi

# 获取根目录参数
ROOTDIR=$1

# 核心部分：查找每个子目录并统计文件数量
find $ROOTDIR -maxdepth 1 -type d | sort | while read dir; do
    count=$(find "$dir" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.gif" -o -iname "*.bmp" -o -iname "*.tiff" \) | wc -l)
    echo "$dir: $count"
done


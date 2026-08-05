## 1. 文件与目录操作

- **`ls`** — 列出目录内容

  ```Bash
  ls -lah   # 以易读格式显示所有文件（含隐藏文件）的详细列表
  ```

- **`cd`** — 切换目录

  ```Bash
  cd /var/log   # 进入 /var/log 目录
  cd ~          # 返回当前用户的家目录
  ```

- **`mkdir`** — 创建目录

  ```Bash
  mkdir -p project/src/bin   # 递归创建多层级目录
  ```

- **`cp`** — 复制文件或目录

  ```Bash
  cp -r dir1 dir2   # 递归复制整个目录 dir1 到 dir2
  ```

- **`mv`** — 移动文件或重命名

  ```Bash
  mv old.txt new.txt     # 文件重命名
  mv file.txt /tmp/      # 将文件移动到 /tmp 目录
  ```

- **`rm`** — 删除文件或目录

  ```Bash
  rm -rf target_folder   # 强制递归删除目录及其内容（慎用）
  ```

## 2. 文本查看与处理

- **`cat` / `less`** — 查看文件内容

  ```Bash
  cat app.log            # 一次性输出文件全部内容
  less app.log           # 分页查看长文件（按 q 退出，/ 键搜索）
  ```

- **`head` / `tail`** — 查看文件头部或尾部

  ```Bash
  tail -f app.log        # 实时追踪日志文件的新新增内容
  tail -n 50 app.log     # 查看最后 50 行
  ```

- **`grep`** — 过滤文本内容

  ```Bash
  grep -i "error" app.log      # 忽略大小写查找包含 error 的行
  grep -rn "main" ./src/       # 在当前 src 目录所有文件中递归查找 main 字符串并显示行号
  ```

- **`find`** — 查找文件

  ```Bash
  find /var/log -name "*.log"  # 在 /var/log 下按名称查找所有 .log 文件
  find . -size +100M           # 查找当前目录下大于 100MB 的文件
  ```

## 3. 系统监控与进程管理

- **`ps`** — 查看静态进程状态

  ```Bash
  ps aux | grep python         # 查找运行中的 python 进程
  ```

- **`top` / `htop`** — 动态监控系统资源

  ```Bash
  top                          # 查看 CPU、内存及实时进程状态（按 q 退出）
  ```

- **`kill` / `killall`** — 终止进程

  ```Bash
  kill -9 1234                 # 强制终止 PID 为 1234 的进程
  ```

- **`df` / `du`** — 磁盘空间统计

  ```Bash
  df -h                        # 以易读格式（GB/MB）查看磁盘分区的挂载与使用情况
  du -sh ./folder              # 统计指定文件夹的总体占用大小
  ```

- **`free`** — 查看内存使用情况

  ```Bash
  free -h                      # 以易读格式显示已用/剩余物理内存与 Swap
  ```

## 4. 压缩与解压

- **`tar`** — 打包与解压缩

  ```Bash
  tar -czvf archive.tar.gz /path/to/dir   # 将目录打包并压缩为 .tar.gz
  tar -xzvf archive.tar.gz                # 解压 .tar.gz 到当前目录
  ```

- **`zip` / `unzip`** — ZIP 格式处理

  ```Bash
  unzip data.zip -d /tmp/                 # 解压到指定目录
  ```
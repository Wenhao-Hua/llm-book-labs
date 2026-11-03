真实预训练模型实验：读者复现说明
版本日期：2026-10-27

一、先选一个目标
本书所有章节机制算例可只用 Python 标准库运行。这里是额外的三个真实模型实验，均在 CPU 上完成，没有 GPU 也能做。
retrieval：加载预训练句向量，比较全部排序和过滤后的证据；接第二、七、八章。
finetune：在小型 BERT 上实际反向传播、更新、保存和重载；接第三、五、十、十三章。
cache：用真实 GPT-2 对照完整前缀与 KV 缓存；接第四、八、十二章。
模型是小型历史检查点，选择它们为了使机制可运行，不表示推荐它们作为当前中文应用的最佳模型。

二、工作位置
在 VS Code 打开随书实验目录，用“终端—新建终端”。终端所在文件夹应同时看得到 real_experiments.py 和 real-experiments 文件夹。
以下命令在该目录执行。不要把它们逐行贴进 Python 的 >>> 提示符。
如果只阅读本书，可跳过安装，直接查看 results 中的原始记录。

三、建立隔离环境
作者实测环境是 Windows、Python 3.13.5、PyTorch 2.8.0+cpu、Transformers 4.56.0。
已有合适环境可以使用它。新建环境的示例为：

  conda create -n llm-book python=3.13.5 -y
  conda activate llm-book
  python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
  python -m pip install -r real-experiments/requirements.txt
  python -m pip check

不要同时用 conda 和 pip 向同一环境安装两个不同的 PyTorch 包。
本次验证是在作者现有环境完成的，没有声称以上命令已在空白虚拟机、所有操作系统和所有 Python 版本逐一执行。requirements 固定直接依赖，相关依赖的已安装版本另存于 environment-lock.txt 供诊断；它不是带散列的跨平台完整安装锁。
如果出现代理连接拒绝，应先检查当前代理地址和服务是否运行，不要用反复更换模型或删除环境代替网络诊断。

四、准备模型文件
模型合计约 660 MB 权重，另有分词器等文件；下载和软件环境还需额外空间。GPT-2 加载与计算的内存高于权重文件大小；本稿未测量进程峰值内存，内存紧张时先单独做检索或微调。
先检查本机缓存：

  python prepare_real_models.py --experiment all

没有文件时，确认接受来源页面的使用条件，再显式联网获取本书固定修订：

  python prepare_real_models.py --experiment all --download

该脚本只下载报告中列出的文件，不加载任意仓库自定义代码。已有文件若散列不符会报错，不会静默替换。
默认缓存为用户目录下 .cache/huggingface/hub。自定义目录时，两条命令都加相同的 --cache-root 路径。
模型来源与版本：
  sentence-transformers/all-MiniLM-L6-v2 @ c9745ed1d9f207416be6d2e6f8de32d1f16199bf
  prajjwal1/bert-tiny @ 6f75de8b60a9f8a2fdf7b69cbd86d9e64bcb3837
  gpt2 @ 607a30d783dfa663caf39e06633721c8d4cfcd7e
模型卡链接和许可标记见送审材料的来源及许可清单。送审包不包含权重。

五、运行与检查
保留作者原始 results，自己的结果写入另一个目录：

  python real_experiments.py --experiment retrieval --output real-experiments/my-results
  python real_experiments.py --experiment finetune --output real-experiments/my-results
  python real_experiments.py --experiment cache --output real-experiments/my-results

一次运行全部可用 --experiment all。脚本设为 CPU、float32、四个线程；固定随机种子并关闭 dropout 后评价。首次加载速度与后续计算速度不同。
运行结束应有 retrieval.json、finetune.json、cache.json、environment.json，以及三个微调检查点目录。结果包含输入散列、版本、原始预测和作用范围。
微调开始时若提示 classifier.weight 和 classifier.bias 新初始化，这是新增分类头的预期提示；训练完成与否应看后续执行和结果记录，不能把这条提示本身当成失败。
离线运行不需要 API 密钥，不调用付费接口，不向外部业务系统写入记录。

六、先复核证据，再看分数
检索：八条资料、六条查询。作者记录的过滤后首位命中为 5/6、前三位命中为 6/6。先找首位失败的查询，再比较旧版与现行版的得分。
微调：24 条训练句、12 条保留句，三种种子各 60 次更新。训练后分别正确 11、11、10 条。请在 holdout_rows 中定位误判，不要只抄准确率。没有用保留成绩选轮次或挑种子。
缓存：单步 logits 在指定容差内一致；三次固定十六 token 的两种计算路径生成相同 token 序列。实际 KV 元素字节数为 2,433,024。时间随硬件和系统负载变化，不要求复制相同秒数。
运行配套审计：

  python audit_real_experiments.py

默认核对作者结果；核对自己的完整三项结果时使用 --results real-experiments/my-results。
若修改了原脚本或数据，散列检查失败是预期提醒：另存一份新实验协议，不能继续把旧记录当成新程序的执行结果。

七、这个实验还不能证明什么
英文手写保留句不是独立抽样的真实用户测试集，也无法排除预训练接触过相似表达。
检索先对合成资料做全量计算再过滤，仅说明候选过滤行为，不证明生产系统中已隔离无权读取的数据。
微调是小模型全参数分类训练，不是 LoRA、QLoRA、DPO 或中文生成微调。
GPT-2 是续写模型；计算结果一致仍可能编造规则。这里没有 GPU 测速、服务并发测试、视觉识别或真实用户测试。
作者未替你完成外部模型资产的商业发行审核；如分发权重、改用真实业务数据或商业部署，应按实际使用方式重新核查。

八、复盘作业
先写预测：哪个检索查询容易失败？命令式表达是否一定要求操作？缓存应改变答案还是只改变计算路径？
运行后逐条对照，把“结果”“解释假设”“下一项对照实验”分开记录。
若要根据错误修改训练数据，原保留集从此承担开发作用。再报告泛化表现时，应另备未参与选择的新资料。

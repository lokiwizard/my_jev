# 第三方来源

`src/my_jev/_upstream/api.py`、`metrics.py`、`prefix_cache.py` 来自 Zefan-Cai/Open-Jev，固定 commit `3308a15ccd7eea1df7a37d6ddc39b023b801ba16`，MIT。许可证保存在同目录 `LICENSE`。这些文件没有修改；工程封装和训练循环另行编写。

Qwen/Qwen3.5-2B 的模型与 tokenizer 由 Hugging Face 固定 revision 的文件哈希确定，保留其 Apache-2.0 条款。可选下载源是 [Qwen 官方 ModelScope 仓库](https://modelscope.cn/models/Qwen/Qwen3.5-2B)，候选字节仍必须符合固定 HF 文件哈希。项目源代码的说明不改变模型或数据的许可证。公开数据的各来源和导出哈希保存在下载后的 manifest 中；没有打包模型、数据或访问凭据到本地代码备份。

原样保留文件的 Git blob 哈希：api.py=`c6d88764806fbd6e5cbb03eb44828cb71a9eb90e`；metrics.py=`d3476a2c7102beb73b25c2bb1c438d02e8077b7a`；prefix_cache.py=`42017988a01bae1e8eeba231bbe89b4d2b831fed`；LICENSE=`d124c8ca600e361f69220666863ea3a2ef1ffed0`。部署时已在服务器比对固定官方 commit 的 tree。

Open-Jev 的完整第三方声明：[THIRD_PARTY_NOTICES.md](https://github.com/Zefan-Cai/Open-Jev/blob/3308a15ccd7eea1df7a37d6ddc39b023b801ba16/THIRD_PARTY_NOTICES.md)。

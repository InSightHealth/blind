# README
模型运行机制：

1. **输入表示**：
   - 首先，我们需要将输入的文本（例如句子或序列）进行嵌入（Embedding）。每个单词或标记都用一个向量表示，这些向量构成了输入矩阵。
   - 然后，我们将位置信息添加到这些嵌入中，以便模型能够理解单词之间的顺序关系。
2. **自注意力计算**：
   - Transformer块的核心是自注意力机制（Self-Attention）。它允许模型在处理输入时关注不同位置的信息。
   - 对于每个单词，我们计算其与其他所有单词之间的注意力分数。这些分数用于加权计算每个单词的表示。
3. **残差连接和层归一化**：
   - 自注意力计算的结果与输入进行残差连接（Residual Connection）。
   - 然后，我们应用层归一化（Layer Normalization）来融合输入和自注意力计算的结果。
4. **前馈神经网络层**：
   - 接下来，我们通过一个前馈神经网络层来进一步处理表示。
   - 这个前馈层通常是一个全连接层，它可以学习更复杂的特征。
5. **多个Transformer块的串联**：
   - 多个Transformer块可以连接在一起，形成编码器或解码器。
   - 编码器用于处理输入序列，解码器用于生成输出序列（例如翻译）。

**下载部署**：如果是第一次使用，会自动联网下载模型到本地，下载完成后即可调用，通过如下代码加载模型`model = AutoModel.from_pretrained(model_name)`，接着加载分词器`tokenizer = AutoTokenizer.from_pretrained(model_name)`。通过`device_map`指定运行设备(GPU/CPU)，选择GPU推理(建议GPU内存>16G)。

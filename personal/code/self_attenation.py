import numpy as np


def self_attention(X, Wq, Wk, Wv, return_weights=False):
    """
    计算自注意力（self-attention）输出。

    支持两种输入形状：
    - 2D 输入：`X` 形状为 (seq, dim)，表示单个序列的特征矩阵。
    - 3D 输入：`X` 形状为 (batch, seq, dim)，表示一批序列的特征矩阵。

    参数：
    - X: 输入张量，2D 或 3D。
    - Wq, Wk, Wv: 投影矩阵，形状通常为 (dim, d_k) 或 (dim, d_v)。
    - return_weights: 若为 True，则返回注意力权重 `weights`，否则只返回 `output`。

    返回：
    - output 或 (output, weights)

    说明：实现与标准缩放点积注意力（scaled dot-product attention）一致，数值稳定性通过减去每行/每批的 max 来增强。
    """

    # 计算查询、键、值的线性投影
    # 对于 2D：Q,K,V 形状为 (seq, d)
    # 对于 3D：Q,K,V 形状为 (batch, seq, d)
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv

    # d_k 为键/查询的维度，用于缩放 scores
    d_k = K.shape[-1]

    if X.ndim == 2:
        # 单序列情况：scores 是 (seq, seq)
        # Q @ K.T 对应每个查询与所有键的点积
        scores = Q @ K.T / np.sqrt(d_k)

        # 为了数值稳定性，减去每行的最大值，再做 softmax
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        weights = exp_scores / exp_scores.sum(axis=1, keepdims=True)

        # 输出是权重与 V 的加权和，形状 (seq, d_v)
        output = weights @ V

    elif X.ndim == 3:
        # 批量情况：Q,K,V 形状为 (batch, seq, d)
        # 需要在每个 batch 内分别计算 scores，得到 (batch, seq, seq)
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)

        # 同样在最后一个维度（seq 维度）上进行数值稳定的 softmax
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

        # 批量矩阵乘法，结果形状 (batch, seq, d_v)
        output = np.matmul(weights, V)

    else:
        # 仅支持 2D/3D 输入，其他维度抛错
        raise ValueError("X must be 2D or 3D")

    if return_weights:
        return output, weights
    return output


def test_self_attention():
    """
    对 `self_attention` 进行两组单元测试：2D 和 3D（batch）。

    测试逻辑：使用固定随机种子生成输入与投影矩阵，计算函数返回的 `output` 和 `weights`，
    然后显式计算 `expected_output = weights @ V`（或批量版本），并断言两者相等。
    """

    # ----------------
    # 2D 测试（单序列）
    # ----------------
    rng = np.random.RandomState(0)
    X = rng.randn(3, 4)          # seq=3, dim=4
    Wq = rng.randn(4, 2)         # 将 dim=4 投影到 d_k=2
    Wk = rng.randn(4, 2)
    Wv = rng.randn(4, 2)

    # 从函数获取 output 与注意力权重
    output, weights = self_attention(X, Wq, Wk, Wv, return_weights=True)

    # 显式计算 V 并用 weights 验证 output
    V = X @ Wv
    expected_output = weights @ V
    assert np.allclose(output, expected_output), "2D 输出与预期不匹配"

    # ----------------
    # 3D 测试（批量）
    # ----------------
    rng = np.random.RandomState(1)
    Xb = rng.randn(5, 3, 4)      # batch=5, seq=3, dim=4
    Wq = rng.randn(4, 2)
    Wk = rng.randn(4, 2)
    Wv = rng.randn(4, 2)

    # 批量调用，得到 (output_b, weights_b)
    output_b, weights_b = self_attention(Xb, Wq, Wk, Wv, return_weights=True)

    # 计算批量的 V 并用权重验证输出
    Vb = Xb @ Wv
    expected_output_b = np.matmul(weights_b, Vb)
    assert np.allclose(output_b, expected_output_b), "3D 输出与预期不匹配"


if __name__ == "__main__":
    test_self_attention()
    print("All tests passed")
    
# demo_app.py
# Jordan標準形に関する問題生成デモ（最新版評価関数を反映した最小版）
#
# ローカル実行:
#   pip install -r requirements.txt
#   streamlit run demo_app.py

import random
from typing import Tuple, List

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


# -----------------------------
# 評価値
# -----------------------------
def evaluate_matrix(A: np.ndarray, best_zero: int, zero_condition: str) -> dict:
    """行列Aの評価値を計算する。

    評価値は「問題としての扱いにくさ」を表す。
    小さいほど，演習問題として扱いやすい行列とみなす。

    最新版 matrix_generator2.py の A_data() に合わせて，
    F_total = F_abs + F_excess + F_sparse + F_zero
    として計算する。
    """
    n = A.shape[0]

    # 0成分の個数
    zero_count = int(np.sum(A == 0))

    # F_abs: 各成分の絶対値の総和
    F_abs = int(np.sum(np.abs(A)))

    # F_excess: |a_ij| が10を超えた分へのペナルティ
    excess_sum = int(np.sum(np.maximum(np.abs(A) - 10, 0)))
    F_excess = 3 * excess_sum

    # F_sparse: 非ゼロ成分が1つだけの行・列へのペナルティ
    sparse_line_count = 0
    for i in range(n):
        if np.count_nonzero(A[i, :]) == 1:
            sparse_line_count += 1
        if np.count_nonzero(A[:, i]) == 1:
            sparse_line_count += 1
    F_sparse = 10 * n * sparse_line_count

    # F_zero: 0成分数に関するペナルティ
    diff = abs(best_zero - zero_count)
    F_zero = diff**3

    if zero_condition == "ちょうど":
        F_zero *= 5
    elif zero_condition == "以上":
        # 目標以上の0があれば弱いペナルティ，不足していれば強いペナルティ
        if best_zero > zero_count:
            F_zero *= 5
    elif zero_condition == "以下":
        # 目標以下の0なら弱いペナルティ，超過していれば強いペナルティ
        if best_zero < zero_count:
            F_zero *= 5
    else:  # 指定なし
        F_zero = 0

    F_total = F_abs + F_excess + F_sparse + F_zero

    return {
        "F_total": int(F_total),
        "F_abs": int(F_abs),
        "F_excess": int(F_excess),
        "F_sparse": int(F_sparse),
        "F_zero": int(F_zero),
        "zero_count": int(zero_count),
        "excess_sum": int(excess_sum),
        "sparse_line_count": int(sparse_line_count),
    }


# -----------------------------
# 基本行列
# -----------------------------
def elementary_matrices(i: int, j: int, n: int) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    """基本行列とその逆行列を返す。

    R(+1): 第j成分に第i成分を加える型
    R(-1): 第j成分から第i成分を引く型
    D(-1): 第i成分を -1 倍
    T    : i と j を入れ替える
    """
    R_plus = np.eye(n, dtype=int)
    R_minus = np.eye(n, dtype=int)
    D = np.eye(n, dtype=int)
    T = np.eye(n, dtype=int)

    R_plus[i, j] = 1
    R_minus[i, j] = -1

    D[i, i] = -1

    T[i, i] = 0
    T[j, j] = 0
    T[i, j] = 1
    T[j, i] = 1

    left = [R_plus, R_minus, D, T]
    right = [np.linalg.inv(M).astype(int) for M in left]
    names = ["R(+1)", "R(-1)", "D(-1)", "T(swap)"]
    return left, right, names


def apply_similarity(A: np.ndarray, L: np.ndarray, R: np.ndarray) -> np.ndarray:
    """A -> L A R を計算する。ここで R = L^{-1}。"""
    return (L @ A @ R).astype(int)


# -----------------------------
# 探索本体
# -----------------------------
class Solver:
    def __init__(
        self,
        J: np.ndarray,
        seed: int,
        best_zero: int,
        zero_condition: str,
        random_steps: int | None = None,
    ):
        self.J = J.astype(int)
        self.A = J.astype(int).copy()
        self.n = J.shape[0]
        self.best_zero = best_zero
        self.zero_condition = zero_condition
        self.rng = random.Random(seed)

        self.Q = np.eye(self.n, dtype=int)
        self.Q_inv = np.eye(self.n, dtype=int)

        self.random_steps = random_steps if random_steps is not None else self.n * 5
        self.A_history = [self.A.copy()]
        self.eval_history = [evaluate_matrix(self.A, self.best_zero, self.zero_condition)]

    def run_random(self) -> np.ndarray:
        """最初にランダムな基本行列で相似変換を行う。"""
        # 最新版と同様に，R(+1), R(-1) を主に選ぶ。
        weights = [45, 45, 9, 1]  # R(+1), R(-1), D, T

        for _ in range(self.random_steps):
            i = self.rng.randrange(self.n)
            j = self.rng.randrange(self.n)
            while i == j:
                j = self.rng.randrange(self.n)

            left, right, _ = elementary_matrices(i, j, self.n)
            k = self.rng.choices([0, 1, 2, 3], weights=weights, k=1)[0]

            L = left[k]
            R = right[k]

            self.A = apply_similarity(self.A, L, R)
            self.Q = L @ self.Q
            self.Q_inv = self.Q_inv @ R

            self.A_history.append(self.A.copy())
            self.eval_history.append(evaluate_matrix(self.A, self.best_zero, self.zero_condition))

        return self.A.copy()

    def run_greedy_search(self, max_iterations: int = 200) -> np.ndarray:
        """評価値が最も下がる R(+1) または R(-1) を貪欲に選ぶ。"""
        current_eval = evaluate_matrix(self.A, self.best_zero, self.zero_condition)["F_total"]

        for _ in range(max_iterations):
            best_eval = current_eval
            best_move = None

            for i in range(self.n):
                for j in range(self.n):
                    if i == j:
                        continue

                    left, right, names = elementary_matrices(i, j, self.n)

                    # 評価値を下げる探索では，最新版と同様に R(+1), R(-1) のみ使う。
                    for k in [0, 1]:
                        candidate_A = apply_similarity(self.A, left[k], right[k])
                        candidate_eval = evaluate_matrix(
                            candidate_A, self.best_zero, self.zero_condition
                        )["F_total"]

                        if candidate_eval < best_eval:
                            best_eval = candidate_eval
                            best_move = (left[k], right[k], candidate_A, names[k], i, j)

            # これ以上評価値が下がらなければ探索終了
            if best_move is None:
                break

            L, R, new_A, _, _, _ = best_move
            self.A = new_A
            self.Q = L @ self.Q
            self.Q_inv = self.Q_inv @ R

            current_eval = best_eval
            self.A_history.append(self.A.copy())
            self.eval_history.append(evaluate_matrix(self.A, self.best_zero, self.zero_condition))

        return self.A.copy()


# -----------------------------
# 表示用
# -----------------------------
def latex_matrix(A: np.ndarray, name: str | None = None) -> str:
    s = ""
    if name:
        s += f"{name} = "
    s += r"\begin{bmatrix}" + "\n"
    for row in A:
        s += " & ".join(str(int(x)) for x in row) + r" \\" + "\n"
    s += r"\end{bmatrix}"
    return s


def input_jordan_matrix(n: int) -> np.ndarray:
    st.write("Jordan標準形 $J$ を入力してください．対角成分と1つ右上の成分のみ入力できます．")

    rows = []
    for i in range(n):
        cols = st.columns(n)
        row = []
        for j in range(n):
            if i == j or j == i + 1:
                default = 0
                if j == i:
                    default = i + 1
                value = cols[j].number_input(
                    f"J[{i+1},{j+1}]",
                    value=int(default),
                    step=1,
                    key=f"J_{i}_{j}",
                )
            else:
                value = 0
                cols[j].markdown("0")
            row.append(int(value))
        rows.append(row)

    return np.array(rows, dtype=int)


def show_evaluation_table(info: dict):
    st.table(
        {
            "項目": [
                "F_abs：各成分の絶対値の総和",
                "F_excess：成分の絶対値が10を超えた分のペナルティ",
                "F_sparse：非ゼロ成分が1つのみの行・列のペナルティ",
                "F_zero：0成分の個数に関するペナルティ",
                "F_total：合計評価値",
                #"0成分の個数",
                #"|成分| が10を超えた分の合計",
                #"非ゼロ成分が1つのみの行・列の個数",
            ],
            "値": [
                info["F_abs"],
                info["F_excess"],
                info["F_sparse"],
                info["F_zero"],
                info["F_total"],
                #info["zero_count"],
                #info["excess_sum"],
                #info["sparse_line_count"],
            ],
        }
    )


def plot_evaluation_history(eval_history: list[dict]):
    x = np.arange(len(eval_history))

    F_abs = np.array([e["F_abs"] for e in eval_history])
    F_excess = np.array([e["F_excess"] for e in eval_history])
    F_sparse = np.array([e["F_sparse"] for e in eval_history])
    F_zero = np.array([e["F_zero"] for e in eval_history])
    F_total = np.array([e["F_total"] for e in eval_history])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, F_abs, marker="o", label="F_abs")
    ax.plot(x, F_excess, marker="s", label="F_excess")
    ax.plot(x, F_sparse, marker="p", label="F_sparse")
    ax.plot(x, F_zero, marker="*", label="F_zero")
    ax.plot(x, F_total, marker="^", linewidth=2, label="F_total")

    ax.set_xlabel("iteration")
    ax.set_ylabel("evaluation value")
    ax.set_title("Evaluation values")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)


# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.set_page_config(page_title="Jordan Matrix Generator Demo", layout="centered")

    st.title("Jordan標準形に関する問題生成デモ")

    st.markdown(
        r"""
このアプリは，指定した Jordan 標準形 $J$ に対して，変換

$$
A = QJQ^{-1}
$$

を用いて，$J$ と相似な整数行列 $A$ を生成するデモです．

生成された行列について，成分の大きさ，10を超える成分，
非ゼロ成分が少ない行・列，0成分の個数に基づく評価値を計算し，
評価値が小さくなるように探索します．
"""
    )

    # pages/1_algorithm.py を作った場合に使う。
    # ファイルが存在しない状態でも demo_app.py 単体では動くように，try で囲む。
    try:
        with st.container(border=True):
            st.markdown("アルゴリズムの概要や評価関数の定義は，別ページにまとめています．")
            st.page_link(
                "pages/Algorithm.py",
                label="アルゴリズムの概要はこちら",
                icon="📘",
            )
    except Exception:
        pass

    st.sidebar.header("設定")

    n = st.sidebar.selectbox("行列サイズ", options=[3, 4], index=0)

    use_seed = st.sidebar.checkbox("乱数シードを固定する", value=False)
    seed = st.sidebar.number_input("乱数シード", min_value=0, value=0, step=1)
    if not use_seed:
        seed = random.randint(0, 10**9)

    best_zero = st.sidebar.number_input(
        "目標とする0成分の個数",
        min_value=0,
        max_value=n * n,
        value=0,
        step=1,
    )

    zero_condition = st.sidebar.radio(
        "0成分の条件",
        options=["ちょうど", "以上", "以下", "指定なし"],
        index=0,
    )

    random_steps = st.sidebar.number_input(
        "初期ランダム変換の回数",
        min_value=0,
        max_value=100,
        value=n * 5,
        step=1,
    )

    with st.expander("Jordan標準形の入力", expanded=True):
        J = input_jordan_matrix(n)

    st.subheader("入力された Jordan 標準形")
    st.latex(latex_matrix(J, "J"))

    if st.button("行列を生成", type="primary"):
        solver = Solver(
            J=J,
            seed=int(seed),
            best_zero=int(best_zero),
            zero_condition=zero_condition,
            random_steps=int(random_steps),
        )

        A_random = solver.run_random()
        A_final = solver.run_greedy_search()
        Q = solver.Q
        Q_inv = solver.Q_inv

        st.markdown("---")
        st.subheader("初期ランダム変換後の行列")
        st.latex(latex_matrix(A_random, "A'"))

        st.subheader("最終的に生成された行列")
        st.latex(latex_matrix(A_final, "A"))

        st.subheader("変換行列")
        st.latex(latex_matrix(Q, "Q"))

        with st.expander("$Q^{-1}$ を表示する"):
            st.latex(latex_matrix(Q_inv, r"Q^{-1}"))

        #with st.expander("相似関係を確認する"):
        #    st.latex(r"A = QJQ^{-1}")
        #    check_A = (Q @ J @ Q_inv).astype(int)
        #    if np.array_equal(check_A, A_final):
        #        st.success("確認成功：A = QJQ^{-1} が成り立っています。")
        #    else:
        #        st.warning("確認注意：A = QJQ^{-1} の確認で差が出ました。")
        #    st.latex(latex_matrix(check_A, r"QJQ^{-1}"))

        st.subheader("評価値")
        eval_info = evaluate_matrix(A_final, int(best_zero), zero_condition)
        show_evaluation_table(eval_info)

        #with st.expander("評価値の変化を表示する"):
        #    plot_evaluation_history(solver.eval_history)

        with st.expander("変形後の行列履歴を表示する"):
            for k, A_k in enumerate(solver.A_history):
                st.latex(latex_matrix(A_k, f"A_{{{k}}}"))


if __name__ == "__main__":
    main()

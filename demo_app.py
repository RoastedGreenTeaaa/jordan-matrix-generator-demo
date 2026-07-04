# Jordan標準形に関する問題生成デモ
#
# ローカル実行:
#   pip install -r requirements.txt
#   streamlit run demo_app.py
#
# マルチページ構成:
#   demo_app.py              # デモ本体
#   pages/Algorithm.py       # アルゴリズム説明ページ

import random
from typing import Tuple, List

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


# -----------------------------
# 評価値
# -----------------------------
def evaluate_matrix(A: np.ndarray, best_zero: int, zero_condition: str) -> dict:
    """行列Aの評価値を計算する．
       評価値は「問題としての扱いにくさ」を表す．
       小さいほど，演習問題として扱いやすい行列とみなす．
          F_total = F_abs + F_excess + F_sparse + F_zero
       として計算
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
    """基本行列とその逆行列を返す．
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
    return (L @ A @ R).astype(int)



# -----------------------------
# 探索本体
# -----------------------------
def _zero_condition_to_eva_type(zero_condition: str) -> int:
    if zero_condition == "ちょうど":
        return 0
    if zero_condition == "以上":
        return 1
    if zero_condition == "以下":
        return 2
    return 3


def A_data(A: np.ndarray, size: int, Best_0: int, eva_type: int):
    sum_eva = 0
    sum_zero = 0
    sum_excess = 0
    count_many_0 = 0
    diff_0 = 0

    for n in range(size):
        for m in range(size):
            if A[n, m] == 0:
                sum_zero += 1
            elif abs(A[n, m]) > 10:
                sum_excess += abs(A[n, m]) - 10

    sum_excess_eva = sum_excess * 3
    A_abs = np.sum(np.abs(A))

    for n in range(size):
        if np.count_nonzero(A[n, :]) == 1:
            count_many_0 += 1
    for m in range(size):
        if np.count_nonzero(A[:, m]) == 1:
            count_many_0 += 1

    many_0_eva = count_many_0 * size * 10

    diff_0 = abs(Best_0 - sum_zero) ** 3
    if eva_type == 0:
        diff_0 *= 5
    elif eva_type == 1:
        if Best_0 <= sum_zero:
            pass
        else:
            diff_0 *= 5
    elif eva_type == 2:
        if Best_0 >= sum_zero:
            pass
        else:
            diff_0 *= 5
    elif eva_type == 3:
        diff_0 = 0

    sum_eva = A_abs + sum_excess_eva + many_0_eva + diff_0

    return sum_eva, sum_zero, A_abs, sum_excess_eva, many_0_eva, diff_0


def set_Q(i: int, j: int, size: int):
    R, R_, D, T = [np.eye(size, dtype=int) for _ in range(4)]

    R[i, j] = 1
    R_[i, j] = -1
    D[i, i] = -1
    T[i, i] = T[j, j] = 0
    T[i, j] = T[j, i] = 1

    Q = [R, R_, D, T]
    q = []

    for m in Q:
        n = np.linalg.inv(m)
        n = np.array(n, dtype=int)
        q.append(n)

    return Q, q


def eva_con(x, A: np.ndarray, size: int, Best_0: int, eva_type: int):
    eva_plus_list = []
    eva_minus_list = []
    eva_place_list = []

    for l in range(size):
        if l != x[0]:
            eva_place_list.append((x[0], l))
        if l != x[1]:
            eva_place_list.append((l, x[1]))

    for k in range(size):
        if k != x[0]:
            row_Q = set_Q(x[0], k, size)

            A_row_plus = row_Q[0][0] @ A @ row_Q[1][0]
            eva_plus_list.append(A_data(A_row_plus, size, Best_0, eva_type)[0])

            A_row_minus = row_Q[0][1] @ A @ row_Q[1][1]
            eva_minus_list.append(A_data(A_row_minus, size, Best_0, eva_type)[0])

        if k != x[1]:
            col_Q = set_Q(k, x[1], size)

            A_col_plus = col_Q[0][0] @ A @ col_Q[1][0]
            eva_plus_list.append(A_data(A_col_plus, size, Best_0, eva_type)[0])

            A_col_minus = col_Q[0][1] @ A @ col_Q[1][1]
            eva_minus_list.append(A_data(A_col_minus, size, Best_0, eva_type)[0])

    return eva_plus_list, eva_minus_list, eva_place_list


class Solver:
    def __init__(
        self,
        J: np.ndarray,
        seed: int,
        best_zero: int,
        zero_condition: str,
        random_steps: int | None = None,
    ):
        self.A = J.astype(int).copy()
        self.size = J.shape[0]
        self.Best_0 = best_zero
        self.eva_type = _zero_condition_to_eva_type(zero_condition)

        self.rng = random.Random(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.Q = np.eye(self.size, dtype=int)
        self.q = np.eye(self.size, dtype=int)

        self.history = {
            "A_list": [self.A.copy()],
            "Q_list": [self.Q.copy()],
            "q_list": [self.q.copy()],
        }

        self.zero_condition = zero_condition
        self.random_steps = self.size * 5  

    @property
    def Q_inv(self) -> np.ndarray:
        return self.q

    @property
    def A_history(self) -> list[np.ndarray]:
        return self.history["A_list"]

    @property
    def eval_history(self) -> list[dict]:
        return [
            evaluate_matrix(A, self.Best_0, self.zero_condition)
            for A in self.history["A_list"]
        ]

    def run_random(self) -> np.ndarray:
        total = 0
        total_last = self.size * 5

        while total < total_last:
            v = self.rng.randint(0, self.size - 1)
            w = self.rng.randint(0, self.size - 1)
            while v == w:
                w = self.rng.randint(0, self.size - 1)

            x = random.choices(
                [0, 1, 2, 3],
                k=1,
                weights=[45, 45, 9, 1],
            )[0]

            left, right = set_Q(v, w, self.size)
            L = left[x]
            R = right[x]

            self.Q = L @ self.Q
            self.q = self.q @ R

            self.A = L @ self.A @ R
            self.history["A_list"].append(self.A.copy())
            self.history["Q_list"].append(self.Q.copy())
            self.history["q_list"].append(self.q.copy())

            total += 1

        return self.A.copy()

    def run_greedy_search(self) -> np.ndarray:
        last_eva = A_data(self.A, self.size, self.Best_0, self.eva_type)[0]

        while True:
            check_eva = last_eva
            check_Q = None
            y = 0

            for o in range(self.size):
                for p in range(self.size):
                    plus_val, minus_val, position = eva_con(
                        [o, p],
                        self.A,
                        self.size,
                        self.Best_0,
                        self.eva_type,
                    )

                    sub_plus = min(plus_val)
                    sub_minus = min(minus_val)

                    if sub_plus < sub_minus:
                        sub_place = plus_val.index(sub_plus)
                        sub_eva = sub_plus
                        sub_Q = position[sub_place]
                        sub_y = 0
                    else:
                        sub_place = minus_val.index(sub_minus)
                        sub_eva = sub_minus
                        sub_Q = position[sub_place]
                        sub_y = 1

                    if sub_eva < check_eva:
                        check_eva = sub_eva
                        check_Q = sub_Q
                        y = sub_y

            if check_eva == last_eva:
                break
            else:
                last_eva = check_eva
                left, right = set_Q(check_Q[0], check_Q[1], self.size)
                L = left[y]
                R = right[y]

                self.Q = L @ self.Q
                self.q = self.q @ R

                self.A = L @ self.A @ R
                self.history["A_list"].append(self.A.copy())
                self.history["Q_list"].append(self.Q.copy())
                self.history["q_list"].append(self.q.copy())

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

    st.title("Jordan標準形を求める問題生成のデモ")

    st.markdown(
        r"""
このアプリは，指定した Jordan 標準形 $J$ に対して，変換

$$
A = QJQ^{-1}
$$

を用いて，$J$ を答えとする整数行列 $A$ を生成するデモです．

生成された行列について，成分の大きさ，$10$ を超える成分，
非ゼロ成分が少ない行・列，$0$ 成分の個数などに基づく評価値を計算し，
評価値が小さくなるように探索します．
"""
    )

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

    zero_condition_label = st.sidebar.radio(
        "0成分の個数について",
        options=[f"ちょうど {best_zero} 個を目指す", 
        f"{best_zero} 個以上を目指す", 
        f"{best_zero} 個以下を目指す", "指定なし"],
        index=0,
    )

    if zero_condition_label == f"ちょうど {best_zero} 個を目指す":
        zero_condition = "ちょうど"
    elif zero_condition_label == f"{best_zero} 個以上を目指す":
        zero_condition = "以上"
    elif zero_condition_label == f"{best_zero} 個以下を目指す":
        zero_condition = "以下"
    else:
        zero_condition = "指定なし"


    #random_steps = st.sidebar.number_input(
    #    "初期ランダム変換の回数",
    #    min_value=n * 5,
    #    max_value=n * 5,
    #    value=n * 5,
    #    step=1,
    #    disabled=True,
    #    help="matrix_generator2.py のアルゴリズムに合わせて n×5 回で固定しています。",
    #)
    random_steps = n * 5

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

        st.subheader("最終結果（問題として出題する行列）")
        st.latex(latex_matrix(A_final, "A"))

        st.subheader("$Q^{-1}AQ = J$ を満たす正則行列")
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

        with st.expander("変形後の履歴を表示する"):
            for k, A_k in enumerate(solver.A_history):
                st.latex(latex_matrix(A_k, f"A_{{{k}}}"))


if __name__ == "__main__":
    main()

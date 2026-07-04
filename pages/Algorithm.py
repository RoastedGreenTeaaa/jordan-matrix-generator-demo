# pages/1_algorithm.py

import streamlit as st

st.set_page_config(page_title="アルゴリズムの説明", layout="centered")

st.title("アルゴリズムの概要")


try:
    with st.container(border=True):
        #st.markdown("行列生成のデモページに戻ることができます。")
        st.page_link(
            "demo_app.py",
            label="デモページに戻る",
            icon="📘",
        )
except Exception:
    pass

st.markdown(r"""
このページでは，本アプリで用いている行列生成アルゴリズムを簡単に説明します．

## 1. 目的

与えられた Jordan 標準形 $J$ に対して，変換

$$
A = QJQ^{-1}
$$

を用いることで，整数行列 $A$ を生成します．
生成された行列 $A$ は，Jordan 標準形 $J$ を求める演習問題として利用することを想定しています．

## 2. 基本的な手順

1. 問題の答えとするJordan 標準形 $J$ を与え，$A(0) = J$ とおきます．

2. 基本行列 $Q$ をランダムに選び，$A(i+1) = Q A(i) Q^{-1}$ 
を一定回数繰り返して，「初期ランダム変換後の行列」を生成します．

3. 得られた行列 $A$ に対して評価値 $F_{\mathrm{total}}(A)$ を計算します．

4. 基本行列による変換候補を調べ，変換後の行列 $A(i+1)$ の評価値がより小さくなる変換を選んで $A$ を更新します．

5. 評価値を小さくする基本変換が見つからなくなった場合，その行列を出力します．

## 3. 評価値

本アプリでは，行列 $A=(a_{ij})$ に対して，次の評価値を用います．
評価値は「問題としての扱いにくさ」を表し，小さいほど良い行列とみなします．

### 成分の絶対値の合計

$$
F_{\mathrm{abs}}(A)
=
\sum_{i,j} |a_{ij}|
$$

### 10を超える成分へのペナルティ

$$
F_{\mathrm{excess}}(A)
=
3\sum_{i,j}\max(|a_{ij}|-10,0)
$$

### 非ゼロ成分が1つだけの行・列へのペナルティ

非ゼロ成分が1つだけの行または列の個数を $N_{\mathrm{sparse}}$，
行列サイズを $n$ とすると，

$$
F_{\mathrm{sparse}}(A)
=
10nN_{\mathrm{sparse}}
$$

### 0成分の個数に関するペナルティ

目標とする $0$ 成分の個数を $B$，実際の0成分の個数を $Z(A)$ とします．

「ちょうど $B$ 個」を目指す場合は，

$$
F_{\mathrm{zero}}(A)
=
5|B-Z(A)|^3
$$

とします．

「$B$ 個以上」を目指す場合は，$0$ 成分が不足しているときに強いペナルティを与えます．

$$
F_{\mathrm{zero}}(A)
=
\begin{cases}
|B-Z(A)|^3, & Z(A)\ge B,\\[5pt]
5|B-Z(A)|^3, & Z(A)<B.
\end{cases}
$$

「$B$ 個以下」を目指す場合は，0成分が多すぎるときに強いペナルティを与えます．

$$
F_{\mathrm{zero}}(A)
=
\begin{cases}
|B-Z(A)|^3, & Z(A)\le B,\\[5pt]
5|B-Z(A)|^3, & Z(A)>B.
\end{cases}
$$

$0$ 成分の個数を指定しない場合は，

$$
F_{\mathrm{zero}}(A)=0
$$

とします．

## 4. 合計評価値

行列 $A$ に対する最終的な評価値は，

$$
F_{\mathrm{total}}(A)
=
F_{\mathrm{abs}}(A)
+
F_{\mathrm{excess}}(A)
+
F_{\mathrm{sparse}}(A)
+
F_{\mathrm{zero}}(A)
$$

です．
""")

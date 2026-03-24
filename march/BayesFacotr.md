# Bayes Factorについて

## ベイズファクターの数学的定義

`test3.py` では、H-form と B-form のモデル比較を SMC で得た周辺尤度
`m(y | M)` を使って行っています。  
ここで

- `y`: 観測データ全体
- `M_H`: H-form モデル
- `M_B`: B-form モデル

とすると、ベイズファクターは

```math
BF_{H/B}
= \frac{p(y \mid M_H)}{p(y \mid M_B)}
= \frac{m(y \mid M_H)}{m(y \mid M_B)}
```

で定義されます。  
分子と分母の周辺尤度は、各モデルのパラメータ `\theta` を積分消去した量です。

```math
m(y \mid M)
= \int p(y \mid \theta, M)\, p(\theta \mid M)\, d\theta
```

つまりベイズファクターは、

- 尤度 `p(y | \theta, M)` だけでなく
- 事前分布 `p(\theta | M)` も含めて

「モデル全体として、観測データをどれだけうまく説明しているか」を比較する指標です。  
`test3.py` の `compute_bayes_factor_smc()` では、SMC が返す
`log_marginal_likelihood` を使って

```math
\log BF_{H/B}
= \log p(y \mid M_H) - \log p(y \mid M_B)
```

を計算しています。さらに

```math
\log_{10} BF_{H/B} = \frac{\log BF_{H/B}}{\log 10}
```

も保存しており、桁感を把握しやすくしています。

## ベイズファクターの統計学的解釈

解釈の基本は符号と大きさです。

- `log_BF > 0`: H-form を支持
- `log_BF < 0`: B-form を支持
- `log_BF = 0`: 両モデルは同程度

例えば

- `log_BF = 2` なら `BF = e^2 ≈ 7.4` で、H-form が約 7 倍支持される
- `log10_BF = 3` なら `BF = 10^3 = 1000` で、H-form が約 1000 倍支持される

という意味になります。

`test3.py` では `abs(log_BF)` の大きさに応じて、次の 4 段階で解釈しています。

- `abs(log_BF) < 1.15`: ほぼ証拠なし
- `1.15 <= abs(log_BF) < 2.3`: 弱い証拠
- `2.3 <= abs(log_BF) < 4.6`: 中程度の証拠
- `4.6 <= abs(log_BF)`: 強い証拠

これは概ね

- `BF < 3`: ほとんど差がない
- `3 <= BF < 10`: 弱い支持
- `10 <= BF < 100`: 中程度の支持
- `BF >= 100`: 強い支持

に対応します。

## 実務上の読み方

ベイズファクターを読むときは、次の順で確認すると実務上の判断がしやすいです。

1. `winner` がどちらかを見る
2. `log_BF` または `log10_BF` の符号で、どちら向きの証拠か確認する
3. `abs(log_BF)` の大きさで、証拠の強さを確認する
4. `summary_H.csv` と `summary_B.csv` を見て、どのパラメータ差がその優位性に効いているかを確認する

注意点として、ベイズファクターは AIC/BIC のような近似的な情報量規準ではなく、
事前分布も含めたモデル比較です。  
そのため、事前分布を大きく変えると `BF` も変わり得ます。  
また、`comparison_ess` はサンプラー効率の比較であり、モデルの優劣そのものを表す量ではありません。モデル選択は `comparison_bayes_factor` を優先して解釈します。
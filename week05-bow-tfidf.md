# BoW and tf-idf

## BoWについて

BoW(Bag of Words)とは、文書をベクトルとして表現する手法。ベクトルにすることで、文書の特徴を掴むことができる。

word2vecと役割は似ているように思う。しかし、実際は全くの別物である。

>word2vecは、大量のテキストデータを解析し、各単語の意味をベクトル表現化する手法。

[word2vec章・冒頭より](./week02-word2vec.md)

BoWは、word2vecよりも遥かに単純である。BoWは、単語の文脈等を考慮せず、機械的にベクトル化する。

### BoWの仕組み

```none
doc = 'Python is programming language. Its name come from TV. The TV program is 'Monty Python's Flying Circus'. It is comedy from BBC.'
```

わかりやすさ~~書きやすさ~~を重視し、上の文書docを例に説明する。

```none
Circus, from, name, language, Python, BBC, is, Flying, programming, It, TV, program, Pythons, The, comedy, come, Its, Monty, 
     0,     0,     0,     1,     1,     0,     1,     0,     1,     0,     0,     0,     0,     0,     0,     0,     0,     0,
     0,     1,     1,     0,     0,     0,     0,     0,     0,     0,     1,     0,     0,     0,     0,     1,     1,     0,
     1,     0,     0,     0,     0,     0,     1,     1,     0,     0,     1,     1,     1,     1,     0,     0,     0,     1,
     0,     1,     0,     0,     0,     1,     1,     0,     0,     1,     0,     0,     0,     0,     1,     0,     0,     0,
     1,     2,     1,     1,     1,     1,     3,     1,     1,     1,     2,     1,     1,     1,     1,     1,     1,     1,
```

1行目は、文書の中に出てきた単語を列挙したものである。

2~4行目は、各文にどの単語が何回出現したかをベクトルに表現したものである。

5行目は、文書レベルのベクトルである。

これらのベクトルを生成すること、また、生成されたベクトルをBoWと呼ぶ。

## tf-idfについて

tf-idfとBoWは密接に関係する。tf-idfは、BoWの欠点を補ったものである。

BoWの欠点とは、「頻繁に出現する」=「特徴的な単語である」とは限らないということである。

英語の文書の場合、I,is,a,theなどの出現頻度が高くなることは必然的であり、そこから新たに分かることはない。

日本語の文書であっても、同じように「である」や「この」「その」の数値が高いベクトルができてしまう。

tf-idfによって、本当に特徴的な単語を見出すことができる。

### tf-idfの仕組み

では、どのようにして特徴的な単語を見つけるのか。

tf-idfでは、文書ごとに単語に対し、重み付けを行うことで特徴的な単語を見つけることができる。

重みづけは、以下の数式に基づいて行われる。

#### tfの計算式

<img src="https://latex.codecogs.com/gif.latex?{tf(t,d)&space;=&space;\frac{n_{t,d}}{\sum_{s&space;\in&space;d}n_{s,d}}}">

<img src="https://latex.codecogs.com/gif.latex?tf(t,d)"> : 文書dにある単語tのtf値

<img src="https://latex.codecogs.com/gif.latex?{n_{t,d}}"> : 単語tの文書dにおける出現回数

<img src="https://latex.codecogs.com/gif.latex?{\sum_{s&space;\in&space;d}n_{s,d}}"> : 文書dのすべての単語の出現回数の総和

#### idfの計算式

<img src="https://latex.codecogs.com/gif.latex?idf(t) = log{\frac{N}{df(t)}}+1">

<img src="https://latex.codecogs.com/gif.latex?idf(t)"> : 単語tのidf値

<img src="https://latex.codecogs.com/gif.latex?N"> : 文書の数

<img src="https://latex.codecogs.com/gif.latex?df(t)"> : 単語tが出現する文書の数

logをとる理由は、文書が多くなり、値そのものが大きくなってしまうと計算ができない、比較ができないため。

規模にかかわらず、値が一定に収まるようにしている。（正則化）

+1は0になることを防いでいる。

なお、scikit-learn等のモジュールでは、若干、計算式が異なる。（とりわけ、分母に+1をしていることが多い）これは、コンピュータ上の計算をしやすくするためである。

#### tfとidfとtf-idfの意味

tf値とidf値の計算式を紹介した。

これらの値は以下の意味を持つ。

<img src="https://latex.codecogs.com/gif.latex?tf(t,d)"> : 単語の文書内出現頻度

<img src="https://latex.codecogs.com/gif.latex?idf(t)"> : dfの逆数。多くの文書に出現する単語は特徴的でない。

tfidfは2つの値の積を取る。

<img src="https://latex.codecogs.com/gif.latex?$$tfidf = tf * df$">

これは、「ある文書には頻繁に出現するものの、どの文書にも頻繁に出現するわけではない単語」を表し、その文書の特徴を示すことができる。

実際に例を挙げてみる。

```python
docA = ['python', 'ruby', 'ruby']
docB = ['python', 'php']
```

* tf('python', docA) = 1/3 = 0.33
* tf('ruby', docA) = 2/3 = 0.66
* tf('python', docB) = 1/2 = 0.5
* tf('php', docB) = 1/2 = 0.5
* idf('python') = log(2/2)+1 = 1
* idf('ruby') = log(2/1)+1 = 1.3
* idf('php') = log(2/1)+1 = 1.3

その結果として、

* tf('python', docA) * idf('python') = 0.33
* tf('ruby', docA) * idf('ruby') = 0.858
* tf('python', docB) * idf('python') = 0.5
* tf('php', docB) * idf('php') = 0.65

となり、両文書に出現する'python'は、'ruby'や'php'よりも低い重みづけがされていることがわかる。

## Exercise

### Ex1

上記で説明したBoWをモジュール等を使わずに実装しなさい。

説明に用いた変数docをデータとして扱うこと。

出力結果の見え方については、特に限定しない。単語とベクトルが正しいことが確認できれば良い。

#### Hint1-1

単語をキーとする辞書を作成すると良い。

### Ex2

上記で説明したtfidfの計算を実際にプログラムにて実行せよ。

docA,docBがどのような長さであっても、良いように作ること。

### Ex3

以下の日本語テキストをtf-idf処理し、特徴的な単語のみを抽出しなさい。ここでは、tf-idfの値が0.7を超えたものを特徴的とする。

<a href="./datasets/ai-wikipedia-text.txt" download="ai-wikipedia-text.txt">ai-wikipedia-text.txt</a>

なお、本テキストは、[人工知能 - Wikipedia](https://ja.wikipedia.org/wiki/%E4%BA%BA%E5%B7%A5%E7%9F%A5%E8%83%BD) からの一部引用である。空行や出典元を示す番号なども記載されている。

不要と思われる行や記号は適宜、取り除くこと。

#### Hint3-1

文や段落で1文書とすると良い。

#### Hint3-2

Mecabにて単語に分ける。また、名詞や動詞に限定する。（裁量は任せるが、助詞などは不要である）

[TOPへ戻る](./index.md)

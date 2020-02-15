# WikiQA

My Qusetion-Answer System：Use Wiki-articles relative to question to generate answers

基于Flask框架，界面如下：

![home](images/home.png)

# Components

```
wikiQA
│  app.py
│  stopwords.txt
│  tools.py
│  v2.db 存储wiki-article的数据库
│  
├─model 存储rank需要的模型（TODO：可用map整合为一个）
│      corpus.pkl
│      doc2idx.pkl
│      sklearn_tfidf.pkl
│      tfidf_matrix.pkl
│      
├─scripts 脚本文件，用于构建数据库以及预处理
│  │  __init__.py
│  │  
│  └─retriever
│          build_db.py
│          build_tdidf.py
│          convert.py
│          prep_file.py
│          prep_text.py
│          __init__.py
│          
├─static 
│  ├─css
│  │      bootstrap.css
│  │      custom.min.css
│  │      
│  └─js
│          jquery-3.4.1.min.js
│          load.js
│          
├─templates
│      index.html
│      
└─wikiqa
    │  __init__.py
    │  
    └─retriever
            doc_db.py
            elasticsearch_ranker.py TODO
            jieba_tfidf_ranker.py 残次
            sklearn_tfidf_ranker.py 可用
            utils.py
            __init__.py
            

```

# Retriver

## 数据获取

在wikidump下载：zhwiki-latest-pages-articles.xml

使用WikiExtractor处理数据

python WikiExtractor.py -b 1024M -o ../extracted --json zhwiki-latest-pages-articles.xml.bz2

## 预处理

使用WikiExtractor提取后，仍然存在许多问题

### 结构性预处理

1. convert.py：使用外部工具opencc.exe进行简繁转换
2. prep_file.py：去除没用的特殊符号（eg.'「『'）

### 内容预处理

prep_text.py，构建数据库时，使用multiprocess中的Pool进行多线程操作，传入参数initargs=(preprocess,)来调用prep_text进行以下内容处理。

1. 删除标题带有“消歧义”的文章

   eg. 北京 (消歧义)[https://zh.wikipedia.org/wiki/%E5%8C%97%E4%BA%AC_(%E6%B6%88%E6%AD%A7%E4%B9%89)](https://zh.wikipedia.org/wiki/北京_(消歧义))

2. 删除标题带有“大纲”“列表”“索引”的文章

   eg.心理学大纲[https://zh.wikipedia.org/wiki/%E5%BF%83%E7%90%86%E5%AD%A6%E5%A4%A7%E7%BA%B2](https://zh.wikipedia.org/wiki/心理学大纲)

   中国大陆报纸列表[https://zh.wikipedia.org/wiki/%E4%B8%AD%E5%9B%BD%E5%A4%A7%E9%99%86%E6%8A%A5%E7%BA%B8%E5%88%97%E8%A1%A8](https://zh.wikipedia.org/wiki/中国大陆报纸列表)

   世界政区索引[https://zh.wikipedia.org/wiki/%E4%B8%96%E7%95%8C%E6%94%BF%E5%8D%80%E7%B4%A2%E5%BC%95](https://zh.wikipedia.org/wiki/世界政區索引)

3. 删除以[可以指：|可以是：|指的可能是：]结尾的页面

   eg.首都省；我的中国梦；中国话

4. 删除text = title 的文章

   eg.中国瀑布

## 构建数据库

使用sqlite3存储文章的title和text

## 构建排序模型

1. 1-gram Tf-idf Ranker

   使用sklearn中的TfidfVectorizer

2. 2-gram

   ```python
   tfidf_model = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=stopwords, ngram_range=(1, 2)).fit(corpus)
   ```

   调整参数ngram_range

# Document Reader

（TODO：通过TOP5 articles获取答案）

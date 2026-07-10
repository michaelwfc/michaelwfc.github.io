## 开源搜索引擎项目

## Lucene系搜索引擎，java开发,包括：

- Lucene
- Solr
- Elasticsearch
- Katta、Compass等都是基于Lucene封装。你可以想象Lucene系有多强大。

- Sphinx搜素引擎，c++开发,简单高性能。

以下重点介绍最常用的开源搜素引擎：Lucene、Solr、Elasticsearch、Sphinx的特点和优劣势选型比较。
开源的向量搜索引擎Milvus。


# Lucene

Lucene的开发语言是Java，也是Java家族中最为出名的一个开源搜索引擎，在Java世界中已经是标准的全文检索程序，它提供了完整的查询引擎和索引引擎，没有中文分词引擎，需要自己去实现，因此用Lucene去做一个搜素引擎需要自己去架构，另外它不支持实时搜索。但是solr和elasticsearch都是基于Lucene封装。
优点：
成熟的解决方案，有很多的成功案例。apache 顶级项目，正在持续快速的进步。庞大而活跃的开发社区，大量的开发人员。它只是一个类库，有足够的定制和优化空间：经过简单定制，就可以满足绝大部分常见的需求；经过优化，可以支持 10亿+ 量级的搜索。

缺点：

需要额外的开发工作。所有的扩展，分布式，可靠性等都需要自己实现；非实时，从建索引到可以搜索中间有一个时间延迟，而当前的“近实时”(Lucene Near Real Time search)搜索方案的可扩展性有待进一步完善

# ElasticSearch
 - [elasticsearch-intro](https://www.elastic.co/guide/en/elasticsearch/reference/current/elasticsearch-intro.html)
 - [GITHUB](https://github.com/elastic/elasticsearch)
  
ElasticSearch是一个基于Lucene构建的开源，分布式，RESTful搜索引擎。设计用于云计算中，能够达到实时搜索，稳定，可靠，快速，安装使用方便。支持通过HTTP使用JSON进行数据索引。

Elasticsearch的优缺点

## 优点
 Elasticsearch是分布式的。不需要其他组件，分发是实时的，被叫做”Push replication”。
 Elasticsearch 完全支持 Apache Lucene 的接近实时的搜索。
 处理多租户（multitenancy）不需要特殊配置，而Solr则需要更多的高级设置。
 Elasticsearch 采用 Gateway 的概念，使得完备份更加简单。
 各节点组成对等的网络结构，某些节点出现故障时会自动分配其他节点代替其进行工作。
## 缺点
 还不够自动（不适合当前新的Index Warmup API）

Elasticsearch 与 Solr 的比较总结
 二者安装都很简单；
 Solr 利用 Zookeeper 进行分布式管理，而 Elasticsearch 自身带有分布式协调管理功能;
 Solr 支持更多格式的数据，而 Elasticsearch 仅支持json文件格式；
 Solr 官方提供的功能更多，而 Elasticsearch 本身更注重于核心功能，高级功能多有第三方插件提供；
 Solr 在传统的搜索应用中表现好于 Elasticsearch，但在处理实时搜索应用时效率明显低于 Elasticsearch。
总之，Solr 是传统搜索应用的有力解决方案，但 Elasticsearch 更适用于新兴的实时搜索应用。


# Slor
-[GITHUB](https://github.com/apache/solr)

Solr是一个高性能，采用Java开发，基于Lucene的全文搜索服务器。

文档通过Http利用XML加到一个搜索集合中。

查询该集合也是通过 http收到一个XML/JSON响应来实现。它的主要特性包括：高效、灵活的缓存功能，垂直搜索功能，高亮显示搜索结果，通过索引复制来提高可用性，提 供一套强大Data Schema来定义字段，类型和设置文本分析，提供基于Web的管理界面等。

优点
 Solr有一个更大、更成熟的用户、开发和贡献者社区。
 支持添加多种格式的索引，如：HTML、PDF、微软 Office 系列软件格式以及 JSON、XML、CSV 等纯文本格式。
 Solr比较成熟、稳定。
 不考虑建索引的同时进行搜索，速度更快。
缺点
 建立索引时，搜索效率下降，实时索引搜索效率不高。





# Sphinx
Sphinx一个基于SQL的全文检索引擎，特别为一些脚本语言（PHP,Python，Perl，Ruby）设计搜索API接口。

Sphinx是一个用C++语言写的开源搜索引擎，也是现在比较主流的搜索引擎之一，在建立索引的事件方面比Lucene快50%，但是索引文件比Lucene要大一倍，因此Sphinx在索引的建立方面是空间换取事件的策略，在检索速度上，和lucene相差不大，但检索精准度方面Lucene要优于Sphinx，另外在加入中文分词引擎难度方面，Lucene要优于Sphinx.其中Sphinx支持实时搜索，使用起来比较简单方便.

Sphinx可以非常容易的与SQL数据库和脚本语言集成。当前系统内置MySQL和PostgreSQL 数据库数据源的支持，也支持从标准输入读取特定格式 的XML数据。通过修改源代码，用户可以自行增加新的数据源（例如：其他类型的DBMS 的原生支持）

2.Sphinx的特点

 高速的建立索引(在当代CPU上，峰值性能可达到10 MB/秒);
 高性能的搜索(在2 – 4GB 的文本数据上，平均每次检索响应时间小于0.1秒);
 可处理海量数据(目前已知可以处理超过100 GB的文本数据, 在单一CPU的系统上可 处理100 M 文档);
 提供了优秀的相关度算法，基于短语相似度和统计（BM25）的复合Ranking方法;
 支持分布式搜索;
 支持短语搜索
 提供文档摘要生成
 可作为MySQL的存储引擎提供搜索服务;
 支持布尔、短语、词语相似度等多种检索模式;
 文档支持多个全文检索字段(最大不超过32个);
 文档支持多个额外的属性信息(例如：分组信息，时间戳等);
 支持断词;

# 开源的Faiss


# TIDB
-[GITHUB](https://github.com/pingcap/tidb)


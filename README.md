# Software Introduction
This Java package corresponds to my research paper as follows:<br />
**Yang Qian, Junming Yin, Yezheng Liu, Yuanchun Jiang. TopicModel4J: A Java Package for Topic Models, to be submitted.**<br />
<br />
This package is about Topic Models for Natural Language Processing (NLP). And **it provides an easy-to-use interface for researchers and data analysts**.<br />

Motivations：I develop this Java package to promote related research about Topic Models for Natural Language Processing (NLP). <br />

When submitting my research paper to a journal, I will publicly release all the source code.<br />

# Data Preprocessing for NLP
This software can do the following text preprocessing:<br />
* (1) Split the sentence to words.<br />
* (2) Lowercase the words and preform lemmatization.<br />
* (3) Remove useless characters, URLs and stop words.<br />

The first example is as follows:<br />
```java
import java.util.ArrayList;
import com.topic.utils.FileUtil;
public class RawDataProcess {
	/**
	 * Functions:
	 * 
	 * (1) Split the sentence to words
	 * (2) Lowercase the words and preform lemmatization
	 * (3) Remove special characters (e.g., #, % and &), URLs and stop words
	 * 
	 * @author: Yang Qian
	 */
	public static void main(String[] args) {
			String line = "http://t.cn/RAPgR4n Artificial intelligence is a known phenomenons "
					+ "in the world today. Its root started to build years "
					+ "ago but the tree started to grow long after. Months ago when our beloved google assistant made her first "
					+ "call to book a haircut appointment in the Google IO event,";
			//get all word for a document
			ArrayList<String> words = new ArrayList<String>();
			//lemmatization using StanfordCoreNLP
			FileUtil.getlema(line, words);
			//remove noise words
			String text = FileUtil.RemoveNoiseWord(words);
			System.out.println(text);
	}
}
```
Running this code, we can obtain the following results:<br />
```java

```







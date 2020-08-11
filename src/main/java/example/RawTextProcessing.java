package example;

import java.io.IOException;
import java.util.ArrayList;
import com.topic.utils.FileUtil;

public class RawTextProcessing {
	/**
	 * Functions:
	 * 
	 * (1) Split the sentence to words
	 * (2) Lowercase the words and preform lemmatization
	 * (3) Remove special characters (e.g., #, % and &), URLs and stop words
	 * 
	 * @author: Yang Qian
	 */
	public static void main(String[] args) throws IOException {
			String line = "http://t.cn/RAPgR4n Artificial intelligence is a known phenomenons "
					+ "in the world today. Its root started to build years";
			//get all word for a document
			ArrayList<String> words = new ArrayList<String>();
			//lemmatization
			FileUtil.getlema(line, words);
			//remove noise words using the default stopwords
			String text = FileUtil.RemoveNoiseWord(words);
			//remove noise words using the customized stopwords
			String text_c = FileUtil.RemoveNoiseWord(words,"data/stopwords");
			//print results
			System.out.println(text);
			System.out.println(text_c);
	}
}

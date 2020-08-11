package example;

import java.io.IOException;
import java.util.ArrayList;
import com.topic.utils.FileUtil;

public class RawFileDataProcessing {
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
		//read raw data from a file
		ArrayList<String> docLines = new ArrayList<String>();
		FileUtil.readLines("data/20ng.txt", docLines, "gbk");
		ArrayList<String> doclinesAfter = new ArrayList<String>();
		for(String line : docLines){
			//get all word for a document
			ArrayList<String> words = new ArrayList<String>();
			//lemmatization
			FileUtil.getlema(line, words);
			//remove noise words
			String text = FileUtil.RemoveNoiseWord(words);
			doclinesAfter.add(text);
		}
		// write the post-treatment data to a new file
		FileUtil.writeLines("data/20ngProcessing.txt", doclinesAfter, "gbk");
	}
}

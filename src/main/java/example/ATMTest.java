package example;

import com.topic.model.AuthorTM;

public class ATMTest {

	public static void main(String args[]) throws Exception{
		AuthorTM atm = new AuthorTM("/data/qianyang/topicmodel4J/citationAu_process", "gbk", ",", 30, 0.1,
				0.01, 800, 20, "/data/qianyang/topicmodel4J/output/");
		atm.MCMCSampling();
//		AuthorTM atm = new AuthorTM("data/citationAu_process", "gbk", ",", 15, 0.1,
//				0.01, 1000, 5, "data/authorTMoutput/");
//		atm.MCMCSampling();
	}
}

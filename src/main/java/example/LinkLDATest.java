package example;

import com.topic.model.LinkLDA;

public class LinkLDATest {

	public static void main(String args[]) throws Exception{
		LinkLDA linklda = new LinkLDA("data/citationLink_process", "gbk", "--", 50, 0.1,
				0.01,0.01, 100, 20, "data/linkldaoutput/");
		linklda.MCMCSampling();
	}
}

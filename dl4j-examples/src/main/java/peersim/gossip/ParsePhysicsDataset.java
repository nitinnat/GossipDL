package peersim.gossip;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;

public class ParsePhysicsDataset {
public static void main(String[] args) {
	try {
		BufferedReader br = new BufferedReader(new FileReader("/home/raghuram/Downloads/data_kddcup04/phy_train.dat"));
		BufferedWriter bw = new BufferedWriter(new FileWriter("/home/raghuram/Downloads/data_kddcup04/phy_train_transformed.dat"));

		String nextLine = "";
		while((nextLine = br.readLine()) != null) {
			String[] row = nextLine.split("\t");
			String outputRow = "";
			if(row[1].equals("0")) {
				outputRow += "-1 ";
			}
			else {
				outputRow += "1 ";
			}
			for(int i=2;i<row.length;i++) {
				double attr = Double.parseDouble(row[i]);
				if(i==21 || i==22 || i==23 || i==45 || i==46 || i==47) {
					if(attr==999) {
						continue;
					}
				}
				if(i==30 || i==56) {
					if(attr==9999) {
						continue;
					}
				}
				int j= i-1;
				outputRow += j+":"+row[i]+" ";
			}
			outputRow.trim();
			bw.write(outputRow+"\n");
		}
		br.close();
		bw.close();
	} catch (Exception e) {
		// TODO Auto-generated catch block
		e.printStackTrace();
	}
}
}

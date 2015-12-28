package pLDA;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;

public class PLDA {

	public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
		// initialize...
		PLDA.test = false;
		PLDA.numDocument = 0;
		PLDA.numTopic = 0;
		PLDA.numWord = 0;
		PLDA.numList = 0;
		PLDA.numIteration = 0;
		PLDA.alpha = 0;
		PLDA.beta = 0;
		PLDA.threshold = 0;
		PLDA.fileData = "";
		PLDA.fileModel = "";
		PLDA.fileMatrix = "";
		PLDA.fileList = "";
		PLDA.filePerplexity = "";
		double baseLikelihood = 0;

		// parse command line...
		for (int i = 0; i < args.length; i++) {
			String command = args[i];
			if (command.equals("--test"))
				PLDA.test = true;
			else if (i == args.length - 1)
				break;
			else if (command.equals("--topic"))
				PLDA.numTopic = Integer.parseInt(args[++i]);
			else if (command.equals("--iteration"))
				PLDA.numIteration = Integer.parseInt(args[++i]);
			else if (command.equals("--alpha"))
				PLDA.alpha = Double.parseDouble(args[++i]);
			else if (command.equals("--beta"))
				PLDA.beta = Double.parseDouble(args[++i]);
			else if (command.equals("--threshold"))
				PLDA.threshold = Double.parseDouble(args[++i]);
			else if (command.equals("--data"))
				PLDA.fileData = args[++i];
			else if (command.equals("--model"))
				PLDA.fileModel = args[++i];
			else if (command.equals("--matrix"))
				PLDA.fileMatrix = args[++i];
			else if (command.equals("--list"))
				PLDA.fileList = args[++i];
			else if (command.equals("--perplexity"))
				PLDA.filePerplexity = args[++i];
		}

		// check parameters...
		if (!PLDA.test && (PLDA.numTopic <= 0))
			PLDA.manual();
		if ((PLDA.numIteration <= 0) || (PLDA.alpha <= 0) || (PLDA.beta <= 0))
			PLDA.manual();

		// check input & output directories...
		if (!new File(PLDA.fileData).exists()) {
			System.out.println("Data file \"" + PLDA.fileData + "\" not exists.");
			return;
		}
		if (PLDA.test) {
			if (!new File(PLDA.fileModel).exists()) {
				System.out.println("Model file \"" + PLDA.fileModel + "\" not exists.");
				return;
			}
		}

		// CHECK POINT
		System.out.println("Initialization success!");
		if (!test)
			System.out.println("Training...");
		else
			System.out.println("Testing...");
		System.out.println("\tIteration number: " + String.valueOf(PLDA.numIteration));
		if (threshold > 0)
			System.out.println("\tLikelihood threshold: " + String.valueOf(PLDA.threshold));
		System.out.println("\tALPHA: " + String.valueOf(alpha));
		System.out.println("\tBETA: " + String.valueOf(PLDA.beta));

		// read data...
		FileReader file = new FileReader(new File(PLDA.fileData));
		BufferedReader reader = new BufferedReader(file);
		ArrayList<Integer> words = new ArrayList<Integer>();
		ArrayList<Integer> lengths = new ArrayList<Integer>();
		String line;
		while ((line = reader.readLine()) != null) {
			String[] tokens = line.split(" ");
			int length = Integer.parseInt(tokens[0]);
			PLDA.numDocument++;
			PLDA.numList += length;
			lengths.add(length);
			for (int j = 1; j <= length; j++) {
				int word = Integer.parseInt(tokens[j]);
				if (word >= PLDA.numWord)
					PLDA.numWord = word + 1;
				words.add(word);
			}
		}
		reader.close();
		file.close();

		// read model if testing...
		if (PLDA.test) {
			file = new FileReader(new File(PLDA.fileModel));
			reader = new BufferedReader(file);
			String[] parameters = reader.readLine().split(" ");
			PLDA.numTopic = Integer.parseInt(parameters[1]);
			int word = Integer.parseInt(parameters[3]);
			if (word > PLDA.numWord)
				PLDA.numWord = word;
		}
		PLDA.topicAlpha = PLDA.numTopic * PLDA.alpha;
		PLDA.wordBeta = PLDA.numWord * PLDA.beta;

		// randomly allocate topic to words and generate files...
		int[] listLength = new int[PLDA.numDocument];										// lengths of documents
		int[] listWord = new int[numList];														// word list
		int[][] matrixTopicWord = PLDA.createMatrix(PLDA.numTopic, PLDA.numWord);	// topic-word matrix
		Random random = new Random();
		Configuration configuration = new Configuration();
		FileSystem system = FileSystem.get(configuration);
		SequenceFile.Writer writerHDFS = SequenceFile.createWriter(system, configuration, PLDA.TEMP_LIST, IntWritable.class, Text.class);
		int offset = 0;
		for (int i = 0; i < PLDA.numDocument; i++) {
			int length = lengths.get(i);
			listLength[i] = length;
			baseLikelihood += length * Math.log(1.0 * length / PLDA.numList);
			String value = "";
			for (int j = 0; j < length; j++) {
				int word = words.get(offset);
				listWord[offset++] = word;
				int topic = random.nextInt(PLDA.numTopic);
				value += " " + String.valueOf(word) + " " + String.valueOf(topic);
				if (!PLDA.test)
					matrixTopicWord[topic][word]++;
			}
			writerHDFS.append(new IntWritable(i), new Text(value.substring(1)));
		}
		IOUtils.closeStream(writerHDFS);
		lengths.clear();
		words.clear();

		// continue to read model if testing, and generate temporary file...
		if (PLDA.test) {
			for (int i = 0; i < PLDA.numTopic; i++) {
				String[] tokens = reader.readLine().split(" ");
				for (int j = 0; j < PLDA.numWord; j++)
					matrixTopicWord[i][j] = Integer.parseInt(tokens[j]);
			}
			reader.close();
			file.close();
		}
		PLDA.writeMatrix(matrixTopicWord, configuration, system, PLDA.TEMP_MODEL);

		// CHECK POINT
		System.out.println("Read input success!");
		System.out.println("\tDocument number: " + String.valueOf(PLDA.numDocument));
		System.out.println("\tTopic number: " + String.valueOf(PLDA.numTopic));
		System.out.println("\tWord number: " + String.valueOf(PLDA.numWord));
		System.out.println("\tTotal words: " + String.valueOf(PLDA.numList));

		// start iterations...
		DistributedCache.addCacheFile(PLDA.TEMP_MODEL.toUri(), configuration);
		double[] likelihoods = new double[PLDA.numIteration];
		for (int r = 0; r < PLDA.numIteration; r++) {
			// map & reduce...
			Job job = new Job(configuration);
			job.setJarByClass(PLDA.class);
			job.setMapperClass(SampleMapper.class);
			job.setReducerClass(SampleReducer.class);
			job.setMapOutputKeyClass(IntWritable.class);
			job.setMapOutputValueClass(Text.class);
			job.setOutputKeyClass(IntWritable.class);
			job.setOutputValueClass(Text.class);
			job.setInputFormatClass(SequenceFileInputFormat.class);
			job.setOutputFormatClass(SequenceFileOutputFormat.class);
			FileInputFormat.addInputPath(job, PLDA.TEMP_LIST);
			system.delete(PLDA.TEMP_OUTPUT, true);
			FileOutputFormat.setOutputPath(job, PLDA.TEMP_OUTPUT);
			job.waitForCompletion(true);

			// parse results...
			int[] listTopic = new int[PLDA.numList];
			int[] vectorTopic = PLDA.createVector(PLDA.numTopic);
			int[][] matrixDocumentTopic = PLDA.createMatrix(PLDA.numDocument, PLDA.numTopic);
			FileStatus[] statuses = system.listStatus(TEMP_OUTPUT);
			for (FileStatus status : statuses) {
				try {
					SequenceFile.Reader readerHDFS = new SequenceFile.Reader(system, status.getPath(), configuration);
					IntWritable key = new IntWritable();
					Text value = new Text();
					while (readerHDFS.next(key, value)) {
						int channel = key.get();
						if (channel < 0) {					// matrix results...
							String[] tokens = value.toString().split(" ");
							for (int i = 2; i < tokens.length; i += 3)
								matrixTopicWord[Integer.parseInt(tokens[i - 2])][Integer.parseInt(tokens[i - 1])] += Integer.parseInt(tokens[i]);
						}
						else {									// list results...
							String[] tokens = value.toString().split(" ");
							offset = 0;
							for (int i = 0; i < channel; i++)
								offset += listLength[i];
							for (int i = 0; i < tokens.length; i++) {
								int topic = Integer.parseInt(tokens[i]);
								listTopic[offset + i] = topic;
								vectorTopic[topic]++;
								matrixDocumentTopic[channel][topic]++;
							}
						}
					}
					IOUtils.closeStream(readerHDFS);
				}
				catch (IOException e) { continue; }
			}

			// compute likelihood...
			offset = 0;
			int document = 0;
			int length = listLength[0];
			double likelihood = 0;
			for (int i = 0; i < PLDA.numList; i++) {
				if (i >= length + offset) {
					offset += length;
					document++;
					length = listLength[document];
				}
				double sum = 0;
				for (int j = 0; j < PLDA.numTopic; j++)
					sum += (PLDA.alpha + matrixDocumentTopic[document][j]) * (PLDA.beta + matrixTopicWord[j][listWord[i]]) / ((PLDA.topicAlpha + length) * (PLDA.wordBeta + vectorTopic[j]));
				likelihood += Math.log(sum);
			}
			likelihoods[r] = likelihood;

			System.out.println("Iteration=" + String.valueOf(r + 1) + " log_likelihood=" + String.valueOf(likelihood + baseLikelihood) + " perplexity=" + String.valueOf(Math.exp(-likelihood / PLDA.numList)));
			if ((r > 0) && (Math.abs(likelihood - likelihoods[r - 1]) < PLDA.threshold))
				PLDA.numIteration = r + 1;

			// output result...
			if (r + 1 == PLDA.numIteration) {
				// write list...
				int index = 0;
				FileWriter writer = new FileWriter(new File(PLDA.fileList));
				for (int i = 0; i < PLDA.numDocument; i++) {
					length = listLength[i];
					writer.write(String.valueOf(length));
					for (int j = 0; j < length; j++)
						writer.write(" " + String.valueOf(listTopic[index++]));
					writer.write("\r\n");
				}
				writer.close();
				PLDA.writeMatrix(matrixDocumentTopic, PLDA.fileMatrix, "Document", "Topic");
				if (!PLDA.test)
					PLDA.writeMatrix(matrixTopicWord, PLDA.fileModel, "Topic", "Word");
				writer = new FileWriter(new File(PLDA.filePerplexity));
				for (int i = 0; i <= r; i++)
					writer.write(String.valueOf(Math.exp(-likelihoods[i] / PLDA.numList)) + "\r\n");
				writer.close();
			}
			else {
				writerHDFS = SequenceFile.createWriter(system, configuration, PLDA.TEMP_LIST, IntWritable.class, Text.class);
				offset = 0;
				for (int i = 0; i < PLDA.numDocument; i++) {
					length = listLength[i];
					String value = "";
					for (int j = 0; j < length; j++, offset++)
						value += " " + String.valueOf(listWord[offset]) + " " + String.valueOf(listTopic[offset]);
					writerHDFS.append(new IntWritable(i), new Text(value.substring(1)));
				}
				IOUtils.closeStream(writerHDFS);
				if (!PLDA.test)
					PLDA.writeMatrix(matrixTopicWord, configuration, system, PLDA.TEMP_MODEL);
			}
		}
	}

	/* function to print a manual */
	private static void manual() {
		System.out.println("Usage: hadoop jar ./PLDA.jar [options]");
		System.out.println("Options:");
		System.out.println("\t--test\t\t\t(Optional) Set program runs in test mode");
		System.out.println("\t--topic [integer]\tNumber of topics, must be positive, useless in test mode");
		System.out.println("\t--iteration [integer]\tMaximum iterations that will be done, must be positive");
		System.out.println("\t--alpha [float]\t\tHyper parameter ALPHA, must be positive");
		System.out.println("\t--beta [float]\t\tHyper parameter BETA, must be positive");
		System.out.println("\t--threshold [float]\t(Optional) The threshold of likelihood difference between iterations that suggests convergence");
		System.out.println("\t--data [string]\t\tDirectory of training or test data");
		System.out.println("\t--model [string]\tDirectory of resulting (in training mode) or existing(in test mode) model");
		System.out.println("\t--list [string]\t\tDirectory of result (list)");
		System.out.println("\t--matrix [string]\tDirectory of result (matrix)");
		System.out.println("\t--perplexity [string]\tDirectory of perplexity in each iteration");
		System.exit(0);
	}

	/* function to write a matrix to file in HDFS */
	private static void writeMatrix(int[][] source, Configuration configuration, FileSystem system, Path target) throws IOException {
		int row = source.length;
		int column = source[0].length;
		SequenceFile.Writer writer = SequenceFile.createWriter(system, configuration, target, IntWritable.class, Text.class);
		for (int i = 0; i < row; i++) {
			String values = "";
			for (int j = 0; j < column; j++) {
				int count = source[i][j];
				if (count != 0)
					values += " " + String.valueOf(j) + " " + String.valueOf(count);
			}
			if (values.isEmpty())
				return;
			writer.append(new IntWritable(i), new Text(values.substring(1)));
		}
		IOUtils.closeStream(writer);
	}

	/* function to write a matrix to file */
	private static void writeMatrix(int[][] source, String target, String nameRow, String nameColumn) throws IOException {
		int row = source.length;
		int column = source[0].length;
		FileWriter writer = new FileWriter(new File(target));
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < column - 1; j++)
				writer.write(String.valueOf(source[i][j]) + " ");
			writer.write(String.valueOf(source[i][column - 1]) + "\r\n");
		}
		writer.close();
	}

	/* function for Mappers and Reducers to write sparse matrix */
	private static String generateSparseMatrix(int[][] source) {
		int row = source.length;
		int column = source[0].length;
		String result = "";
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < column; j++) {
				int count = source[i][j];
				if (count != 0)
					result += " " + String.valueOf(i) + " " + String.valueOf(j) + " " + String.valueOf(count);
			}
		}
		if (result.isEmpty())
			return "";
		return result.substring(1);
	}

	/* function to create a vector */
	private static int[] createVector(int dimension) {
		int[] result = new int[dimension];
		for (int i = 0; i < dimension; i++)
			result[i] = 0;
		return result;
	}

	/* function to create a matrix */
	private static int[][] createMatrix(int row, int column) {
		int[][] result = new int[row][column];
		for (int i = 0; i < row; i++)
			for (int j = 0; j < column; j++)
				result[i][j] = 0;
		return result;
	}

	/* Mapper in iterations */
	private static class SampleMapper extends Mapper<IntWritable, Text, IntWritable, Text> {

		/* function to initialize and read topic-word matrix */
		public void setup(Context context) throws FileNotFoundException, IOException {
			this.vectorTopic = PLDA.createVector(PLDA.numTopic);
			this.matrixTopicWord = PLDA.createMatrix(PLDA.numTopic, PLDA.numWord);

			// read current model...
			Configuration configuration = context.getConfiguration();
			FileSystem system = FileSystem.get(configuration);
			SequenceFile.Reader reader = new SequenceFile.Reader(system, DistributedCache.getLocalCacheFiles(configuration)[0], configuration);
			IntWritable key = new IntWritable();
			Text value = new Text();
			while (reader.next(key, value)) {
				int topic = key.get();
				String[] values = value.toString().split(" ");
				for (int i = 1; i < values.length; i += 2) {
					int count = Integer.parseInt(values[i]);
					this.matrixTopicWord[topic][Integer.parseInt(values[i - 1])] = count;
					this.vectorTopic[topic] += count;
				}
			}
			IOUtils.closeStream(reader);
		}

		/* function to sample and reallocation topic of each word */
		public void map(IntWritable key, Text value, Context context) throws IOException, InterruptedException {
			this.matrixDocumentTopic = PLDA.createVector(PLDA.numTopic);

			// read current list...
			String[] tokens = value.toString().split(" ");
			int length = tokens.length >> 1;
			int[] listWord = new int[length];
			int[] listTopic = new int[length];
			for (int i = 0; i < length; i++) {
				listWord[i] = Integer.parseInt(tokens[i << 1]);
				int topic = Integer.parseInt(tokens[(i << 1) + 1]);
				listTopic[i] = topic;
				this.matrixDocumentTopic[topic]++;
			}

			// generate an order...
			Random random = new Random();
			int[] order = new int[length];
			for (int i = 0; i < length; i++)
				order[i] = i;
			for (int i = 0; i < length; i++) {
				int r = i + random.nextInt(length - i);
				order[i] ^= order[r];
				order[r] ^= order[i];
				order[i] ^= order[r];
			}

			// sample and reallocation...
			double[] posteriors = new double[PLDA.numTopic];
			int[][] difference = PLDA.createMatrix(PLDA.numTopic, PLDA.numWord);
			for (int i = 0; i < length; i++) {
				int index = order[i];
				int word = listWord[index];
				int topic = listTopic[index];
				this.matrixDocumentTopic[topic]--;
				if (!PLDA.test) {
					this.matrixTopicWord[topic][word]--;
					this.vectorTopic[topic]--;
					difference[topic][word]--;
					if (this.matrixTopicWord[topic][word] < 0)
						System.out.println(String.valueOf(topic) + " " + String.valueOf(word) + String.valueOf(this.matrixTopicWord[topic][word]));
				}

				// calculate posterior of topics...
				double sum = 0;
				for (int j = 0; j < PLDA.numTopic; j++) {
					posteriors[j] = (PLDA.alpha + this.matrixDocumentTopic[j]) * (PLDA.beta + this.matrixTopicWord[j][word]) / (PLDA.wordBeta + this.vectorTopic[j]);
					sum += posteriors[j];
				}

				// sample a topic...
				sum *= random.nextDouble();
				topic = -1;
				while(++topic < PLDA.numTopic - 1) {
					if (sum > posteriors[topic])
						sum -= posteriors[topic];
					else
						break;
				}

				listTopic[index] = topic;
				this.matrixDocumentTopic[topic]++;
				if (!PLDA.test) {
					this.matrixTopicWord[topic][word]++;
					this.vectorTopic[topic]++;
					difference[topic][word]++;
				}
			}

			// output list and difference matrix...
			String result = "";
			for (int i = 0; i < length; i++)
				result += " " + String.valueOf(listTopic[i]);
			context.write(key, new Text(result.substring(1)));
			context.write(new IntWritable(-1), new Text(PLDA.generateSparseMatrix(difference)));
		}

		private int[] vectorTopic;
		private int[] matrixDocumentTopic;
		private int[][] matrixTopicWord;
	}

	/* Reducer in iterations */
	private static class SampleReducer extends Reducer<IntWritable, Text, IntWritable, Text> {

		/* function to aggregate changes of matrix and output new list */
		public void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
			int channel = key.get();
			if (channel < 0) {	// negative key indicates topic-word matrix...
				int[][] difference = PLDA.createMatrix(PLDA.numTopic, PLDA.numWord);
				for (Text value : values) {
					String[] tokens = value.toString().split(" ");
					for (int i = 2; i < tokens.length; i += 3)
						difference[Integer.parseInt(tokens[i - 2])][Integer.parseInt(tokens[i - 1])] += Integer.parseInt(tokens[i]);
				}
				context.write(key, new Text(PLDA.generateSparseMatrix(difference)));
			}
			else						// otherwise indicates list...
				context.write(key, values.iterator().next());
		}
	}

	private static boolean test;												// [argument] test on existing model?
	private static int numDocument;											// number of documents
	private static int numTopic;												// [argument] number of topics
	private static int numWord;												// number of words in vocabulary
	private static int numList;												// number of words in corpus
	private static int numIteration;											// [argument] number of iterations
	private static double alpha;												// [argument] hyper parameter: ALPHA
	private static double beta;												// [argument] hyper parameter: BETA
	private static double topicAlpha;										// ALPHA * number of topics
	private static double wordBeta;											// BETA * number of words
	private static double threshold;											// [argument] threshold of convergence
	private static String fileData;											// [argument] directory of training/test data
	private static String fileModel;											// [argument] directory of model (written when trained and read when tested)
	private static String fileMatrix;										// [argument] directory of matrix in result
	private static String fileList;											// [argument] directory of list in result
	private static String filePerplexity;									// [argument] directory of perplexity in each iteration
	private static final Path TEMP_MODEL = new Path("/model");		// temporary model file in HDFS
	private static final Path TEMP_LIST = new Path("/list");			// temporary list file in HDFS
	private static final Path TEMP_OUTPUT = new Path("/output");	// temporary output of reducers in HDFS
}

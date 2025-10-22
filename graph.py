


from typing import Annotated
from typing import List
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END


import chunker

#ranked by performance/quality
mod=["sentence-transformers/embeddinggemma-300m-medical","sentence-transformers/all-MiniLM-L6-v2","sentence-transformers/msmarco-MiniLM-L12-v3","sentence-transformers/all-MiniLM-L12-v2","sentence-transformers/all-mpnet-base-v2"]
    
chk = chunker.getModel(mod[0],minChunkSize=800)

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    #messages: Annotated[list, add_messages]
    text:  str
    chunks: List[str]
    metaData: dict[str,any]




def getData(state: State):
    #user inputs promt and it gets the data/docs    OR  it gets data from pubMed stores it and then we retrieve it

    #return {"text": getData() ,"metaData": getMetaData()}
    pass

def chunk(state: State):
    return {"chunks": chunker.splitText(chk,state["text"])}


def storeDB(state: State):
    #return {"text": state["text"] ,"chunks": chunker.splitText(chk,state["text"])}
    pass


def cluster(state: State):
    #return {"text": state["text"] ,"chunks": chunker.splitText(chk,state["text"])}
    pass

def labelC(state: State):
    #return {"text": state["text"] ,"chunks": chunker.splitText(chk,state["text"])}
    pass

def retrieveChunks(state: State):
    #return {"text": state["text"] ,"chunks": chunker.splitText(chk,state["text"])}
    pass

def genPrompt(state: State):
    #return {"text": state["text"] ,"chunks": chunker.splitText(chk,state["text"])}
    pass


def evaluateQuery(state: State):
    #return {"text": state["text"] ,"chunks": chunker.splitText(chk,state["text"])}
    pass

if __name__ == '__main__':
    
    graph_builder = StateGraph(State)

    graph_builder.add_edge(START,"chunker")


    #query processing, retrieval, reranking, prompt assembly, LLM calls
    

    #------sprint1------

    #userS1
    #graph_builder.add_edge("dataExtracter",getData)

    #userS2
    graph_builder.add_node("chunker",chunk)

    #------sprint2------


    #userS3
    #graph_builder.add_edge("storeChunk",storeDB)

    #userS4
    #graph_builder.add_edge("cluster",cluster)

    #userS4
    #graph_builder.add_edge("retrieveCitations",retrieveChunks)

    #userS4
    #graph_builder.add_edge("labelCluster",labelC)

    #------sprint3------


    #userS5
    #graph_builder.add_edge("generateQestionPrompt", genPrompt)
    #graph_builder.add_edge("generateAnswerPrompt", genPrompt)
    #graph_builder.add_edge("generateSummary", genPrompt)

    #------sprint3------


    #userS7
    #graph_builder.add_edge("evaluateQuery",evaluateQuery)



    graph_builder.add_edge("chunker",END)

    graph=graph_builder.compile()

    #source: https://ieeexplore.ieee.org/document/5664887
    fullText="Abstract—Intrusion Detection System (IDS) is the key technology to ensure the security of dynamic systems. We employ a sequential pattern mining approach to discover significant system call sequences to prevent malicious attacks. To reduce the computing time of generating meaningful rules, we design a weighted suffix tree structure to detect intrusive events on the fly. The experimental results show our method can substantially enhance the accuracy and efficiency of IDS. I. INTRODUCTION An intrusion is defined as ”any set of actions that attempt to compromise the integrity, confidentiality or availability of a resource” [5]. Many intrusion prevention techniques, such as user authentication, information protection and programming errors avoidance, have been used to protect information systems from being intruded. With the increasing usage rate of computer users and the Internet, many malicious users and sophisticated hackers attempt to attack computer systems and grab private information. Intrusion detection system, therefore, has become an important solution to enhance the security of information systems. An IDS can detect and report intrusions to an operator but not prevent it. It can be divided into two types: centralized IDS which is performed on a single machine, and distributed IDS which is performed on multiple machines. Furthermore, IDS can be host-based or network-based; the former monitors activities on a single computer and the latter monitors activities over a network. All IDSs consist of three parts: data collection, data classification and data reporting. The data collection tasks collect several types of data: ∙ Network data ∙ System calls of operating system ∙ Command line of operating system ∙ Codes within applications ∙ All characters transmitted ∙ Keystrokes The network data comprises many features which can be analyzed; however, it is always encrypted for information privacy. To analyze the application is difficult because most of the source codes are not released. Unlike the aforementioned data types, the collection of system calls is not affected by data encryption, programming languages and operating systems. We can get system calls easily by monitoring operating system. The advantages of utilizing system calls as dataset, therefore, motivate us to develop a new IDS based on it. The data classification tasks can be divided into two categories [2], depending on whether researchers look for known intrusion signatures (misuse intrusion detection) [8][9][10][11] or anomalous behavior (anomaly intrusion detection) [19][27]. A misuse-based IDS requires prior knowledge of the intrusion, and they use these intrusion signatures to detect the occurrence of intrusion. By contrast, an anomaly-based IDS assumes that the intrusion behavior is unknown, but it is different from the behaviors of normal usage. The data reporting tasks inform system administrators when the anomalous or intrusive behaviors happened. In this paper, we design a weighted suffix tree structure together with sequential pattern mining method to discover meaningful sequential intrusion patterns for protecting malicious attacks in information systems. II. RELATED WORK A. Data Mining Data mining, also called Knowledge-Discovery in Database (KDD), is the process of automatically discovering unknown, implicit and meaningful patterns from large volumes of data. Many previous researchers applied typical data mining approaches to reveal specialized abnormal patterns [4][6][13]. Lee et al. used various kinds of mining-based model to improve IDS [12][15][16][17][18][22]. Li and Pan [14] proposed a 𝜙-association rule mining model based on FP-tree structure to improve the effectiveness of IDS. Xu and Gu [26] utilized the Apriori algorithm [1] to mine malicious attacks. The aforementioned research focus on finding nonsequential system call patterns instead of considering the sequence between system calls. In this paper, we concern the sequence between ordered system calls and extract significant sequential abnormal patterns for IDS. B. Suffix Tree A suffix tree is an edge-labeled compact tree with n leaves introduced by Weiner [24]. Many researchers proposed simplified methods to suffix tree construction [20][23]. Now, we illustrate a basic suffix tree structure through an example. Without loss of generality, we assume that the string S of length 5 is {03, 01, 03, 01, 15} and consequently were the suffixes of S which are {03, 01, 03, 01, 15}, {01, 03, 01, 15}, {03, 01, 15}, {01, 15}, {15}. We could find all suffixes of S showed in Fig. 1. III. SYSTEM ARCHITECTURE The overall system framework is developed to support a sequential mining-based IDS. It can strengthen the security of information systems, and it allows users to prevent malicious attacks. The system architecture is shown in Fig. 2, and it mainly consists of four parts: weighted suffix tree construction model, sequential pattern mining model, rule set pool and the decision engine. Functionalities of each component are described as follows. Weighted suffix tree construction model is used to store all sequences of system calls effectively and efficiently for further computation. We develop a new suffix tree structure to store the occurrences of all system calls which can avoid repetitively reconstructing suffix tree. After constructing weighted suffix tree, the sequential pattern mining model generates time-ordered system call patterns, and moves these possible malicious patterns into a rule set pool. We utilize the rule set pool to create a training model in the decision engine, therefore, once a malicious attack occurs, we can detect and report the intrusive event immediately. IV. METHODOLOGY A. Problem Formulation Definition 1: A set of system calls which are arranged in time order is called a ”system call sequence” [7][15]. Fig. 3. Weighted suffix tree after inserting < 03(1), 01(1), 15(1) >. Definition 2: A list of system calls issued by a single process from the beginning of its execution to the end is called ”trace” [25]. B. Weighted Suffix Tree We propose a novel weighted suffix tree structure which is an extended suffix tree structure to store sequential system calls efficiently and to prevent repetitively suffix tree construction. The weighted suffix tree structure is capable of simultaneously storing multiple system call sequences. After the construction of the weighted suffix tree, we could calculate the frequent episodes through the statistics information of the nodes. We give an example to illustrate the process of weighted suffix tree construction. Assuming we have two sequences 𝑆1 and 𝑆2 in dataset D. D = {𝑆1, 𝑆2}, 𝑆1 = {03, 01, 03, 01, 15} and 𝑆2 = {01, 01, 03, 01}. The weighted suffix tree is constructed as follows: 1. The root of tree is created and labeled with ”null”. 2. Inserting each sequence into the suffix tree, e.g. 𝑆1: < 03(1), 01(1), 03(1), 01(1), 15(1) > leads to the construction of the first branch of the tree, and the suffix of 𝑆1: < 01(1), 03(1), 01(1), 15(1) > leads to the construction of the second branch of the tree. 3. If a sequence < 03(1), 01(1), 15(1) > shares a common prefix < 03(1), 01(1) > with the existed branch < 03(1), 01(1), 03(1), 01(1), 15(1) >, the weight of each node with a common prefix is incremented by 1, and a new node 15(1) is created as a child of node 01(1). The result after inserting < 03(1), 01(1), 15(1) > is shown in Fig. 3. 4. Repeating the above steps, we can construct a weighted suffix tree. Fig. 4 shows the result after processing sequence 𝑆1 and 𝑆2. We record the traversed number of each node when we built the tree. For example, the rightmost node of weighted suffix tree in Fig. 4 is noted as 15(1). The number 15 means the system call ID of that node, and the number 1 surrounded by parentheses means the traversed frequency of that node. The weighted suffix tree construction algorithm is described in Fig. 5 C. Frequent Episodes Mining Mannila et al. proposed a frequent episodes mining algorithm [21] which is an extension of association rules. A frequent episode is a set of items that occur frequently within a time window of a specified length. We give a brief example to explain the rule of frequent episode used in our research below: 03, 01 → 03[𝑠𝑢𝑝 = 20%, 𝑐𝑜𝑛𝑓 = 50%, 𝑙𝑒𝑛𝑔𝑡ℎ = 3] A support of 20% for frequent episode rules means that 20% of all sequences with length 3 under analysis show that system call 03, 01, 03 are used in time order and appeared together. A confidence of 50% means that 50% of the system calls that system call 03, 01 appeared in order and system call 03 also appeared subsequently. We show the definition of support and confidence used in this paper below: 𝑆𝑢𝑝 = Fig. 6. Frequent episodes mining algorithm. 𝑇 ℎ𝑒 𝑤𝑒𝑖𝑔ℎ𝑡𝑒𝑑 𝑐𝑜𝑢𝑛𝑡 𝑜𝑓 𝑡ℎ𝑒 𝑙𝑎𝑠𝑡 𝑛𝑜𝑑𝑒 𝑜𝑓 𝑡ℎ𝑒 𝑠𝑒𝑞𝑢𝑒𝑛𝑐𝑒 𝑆𝑢𝑚 𝑜𝑓 𝑡ℎ𝑒 𝑤𝑒𝑖𝑔ℎ𝑡𝑒𝑑 𝑐𝑜𝑢𝑛𝑡 𝑜𝑓 𝑡ℎ𝑒 𝑙𝑎𝑠𝑡 𝑛𝑜𝑑𝑒 𝑜𝑓 𝑎𝑙𝑙 𝑠𝑒𝑞𝑢𝑒𝑛𝑐𝑒𝑠 𝐶𝑜𝑛𝑓 = 𝑇 ℎ𝑒 𝑤𝑒𝑖𝑔ℎ𝑡𝑒𝑑 𝑐𝑜𝑢𝑛𝑡 𝑜𝑓 𝑡ℎ𝑒 𝑙𝑎𝑠𝑡 𝑛𝑜𝑑𝑒 𝑜𝑓 𝑡ℎ𝑒 𝑠𝑒𝑞𝑢𝑒𝑛𝑐𝑒 𝑆𝑢𝑚 𝑜𝑓 𝑡ℎ𝑒 𝑤𝑒𝑖𝑔ℎ𝑡𝑒𝑑 𝑐𝑜𝑢𝑛𝑡 𝑎𝑛𝑑 𝑠𝑖𝑏𝑙𝑖𝑛𝑔𝑠 𝑜𝑓 𝑡ℎ𝑒 𝑙𝑎𝑠𝑡 𝑛𝑜𝑑𝑒 The frequent episodes mining algorithm described in Fig. 6 is exploited to extract the rule sets with different length after the weighted suffix tree has been constructed. In the past, much research had troubles choosing the most appropriate length of sequential system calls for mining. They must run simulations to get the appropriate length of system call sequences, and repeatedly read whole traces when they need to collect other rule sets with other lengths of sequential system calls [7][15]. However, through our proposed weighted suffix tree, we can read the whole traces once instead of reading them many times while discovering different length of system call sequences. D. Intrusion Detection Model The t-stide method is a well known sequential-based technique proposed by C. Warrender et al. [25]. We apply t-stide method to verify whether intrusive events occur or not, and set a threshold to prune rare sequences which are regarded as abnormal patterns. We choose sup and conf defined by above section as threshold conditions in t-stide method. We, then, collect all frequent sequences of length k which are greater than or equal to the threshold and store them together for further usage. Finally, the decision engine compares the traces with the rule set, and reports the abnormal events to the user interface when mismatch rate exceeds the threshold. V. EXPERIMENTAL RESULTS We utilize different kinds of datasets provided by University of New Mexico [3] to evaluate the execution time and accurate rate of our methodology. We use 80% of normal traces to generate rule sets with different lengths, and the rest 20% of normal traces are used for testing. There are three sscp (sunsendmail) attack traces, two decode attack traces, and five error condition-forwarding loops attack traces for testing. Fig. 7 and Fig. 8 show the performance between the weighted suffix tree and non-weighted suffix tree of maximum sequences with different length. Fig. 7 provides data on the construction time, and it is apparent from the information supplied that the construction time of the weighted suffix tree is significantly less than non-weighted suffix tree. The  results reflected in Fig. 7 indicate that the weighted suffix tree structure can store all different length of sequences, and it can apparently reduce the cost of reconstructing a suffix tree. The mining time of generating abnormal patterns is given in Fig. 8, and it highlights differences between the two treebased structures. As observed in Fig. 8, the performance of using weighted suffix tree is better than non-weighted suffix tree. A glance at the two tables provided reveals the experimental results using the rules with length 5 and length 7. The first row of Table I and Table II is the experimental results running with the method proposed by Warrender et al. [25]. The column of normal abn% means the mismatch rate of normal traces. We also record the mismatch rate of the sscp (sunsendmailcp), the decode, and the fwd (forwarding loops). In Table I, we use the method proposed by [25] to generate 660 rules, and the mismatch rate of normal traces, sscp, decode, and fwd are 3.3%, 16.5%, 5.9%, and 13.9%, respectively. Hence the minimum gap of mismatch rate between normal traces and abnormal traces is 2.6%. After we simulate the experiments with different support and confidence thresholds, we observe that the number of rules generated by our methodology is less than Warrender et al. [25]. The phenomenon indicates that the decision engine of IDS is faster than it does before [25]. By contrast, the mismatch rate of normal traces is higher than the method of [25]. On the other hand, the mismatch rate of abnormal traces will also be higher than the method of [25]. From Table I and Table II, it is evident that the gap generated by our methodology is larger than original method, and it help us to prove that the decision engine can explicitly recognize which trace is normal or abnormal. VI. CONCLUSIONS IDS is a technique which is proposed to improve the security of computer system, and it needs to process a lot of data sets to carry on analysis. This characteristic makes the application of data mining an important role in IDS. Much research apply data mining to support IDS, but they just only find frequent patterns without concerning the order between system calls. In this paper, we develop a feasible suffix treebased data structure to mine time-ordered patterns. From the experimental results, it is evident that our methodology can reduce the construction time of multiple system call sequences, and shorten the mining time of generating sequential patterns. The experimental results also show that the gap between the mismatch rate of normal traces and the mismatch rate of abnormal traces is large. It means that our methodology can help IDS to definitely recognize intrusive events. ACKNOWLEDGMENT This research was sponsored by National Science Council, Taiwan under the grant 99-2219-E-002-021."
    
    temp=graph.invoke({"text": fullText})

    print(temp["chunks"][0])

    #for event in graph.stream({"text": fullText,"chunks": ["empty"],"monkeysPaw": 0}):
    #        for value in event.values():
    #            print("Assistant:", value["chunks"][0])





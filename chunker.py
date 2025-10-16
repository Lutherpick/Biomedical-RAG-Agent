from langchain_experimental import text_splitter

from langchain_huggingface import HuggingFaceEmbeddings
from typing import List

from huggingface_hub import snapshot_download
from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter




#fixed size chunking
def getFixedChunker(chunk_size,chunkCountSymbol=' '):
    splitter = CharacterTextSplitter(chunk_size = chunk_size, chunk_overlap=0, separator=chunkCountSymbol, strip_whitespace=False)
    #splitter.split_text("money money 1 2 3 4 5 6 7")
    return splitter

#semmantic splitting
def loadModel(modelName,modelPath) -> text_splitter.SemanticChunker:
    
    model_kwargs = {"device": "cpu","trust_remote_code":True}
    encode_kwargs = {"normalize_embeddings": False}
    hf = HuggingFaceEmbeddings(
        model_name=(modelPath+modelName),
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    #passive use of model. CODE LOADS MODEL AND USES IT
    #splitter=text_splitter.SemanticChunker(embeddings=hf) 
    model=text_splitter.SemanticChunker(embeddings=hf,buffer_size=1, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=75)

    return model

def downloadModel(modelName,folderPath="./models/"):

    dir =Path(folderPath+modelName)
    #hf_hub_download(repo_id=modelName,local_dir=folderPath)
    if not dir.is_dir():
        snapshot_download(repo_id=modelName,local_dir=(folderPath + modelName))

def getModel(modelName):
    downloadModel(modelName)
    mod=loadModel(modelName=modelName,modelPath="./models/")
    return mod


#for qdrant
def getEmbeddings(modelPath,modelName):
    model_kwargs = {"device": "cpu","trust_remote_code":True}
    encode_kwargs = {"normalize_embeddings": False}
    hf = HuggingFaceEmbeddings(
        model_name=(modelPath+modelName),
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

def test():
    #Sentence Transformers and Universal Sentence Encoders are two different things!: https://milvus.io/ai-quick-reference/what-is-the-difference-between-sentence-transformers-and-other-sentence-embedding-methods-like-the-universal-sentence-encoder
    #encoders and embedders: https://medium.com/@sharifghafforov00/understanding-encoders-and-embeddings-in-large-language-models-llms-1e81101b2f87
    
    #you can also load models and save them like this. not recommended in my opinion. rather do it through cli hf tool
    #model =SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    #model.save("/home/luke-weiss/dev/team_project/all-MiniLM-L6-v2",model_name="all-MiniLM-L6-v2")
    #model.encode(fullText) #active use of model. YOU USE THE MODEL!




    #testing - remove later
    #hf download --local-dir all-MiniLM-L6-v2 sentence-transformers/all-MiniLM-L6-v2
    #hf download --local-dir all-mpnet-base-v2 sentence-transformers/all-mpnet-base-v2
    #hf download --local-dir all-MiniLM-L12-v2 sentence-transformers/all-MiniLM-L12-v2


    with open("fullText.txt") as f:
        fullText = f.read()
    
    modelsPaths=["sentence-transformers/all-MiniLM-L6-v2","/home/luke-weiss/dev/team_project/all-MiniLM-L6-v2","/home/luke-weiss/dev/team_project/all-MiniLM-L12-v2","/home/luke-weiss/dev/team_project/all-mpnet-base-v2"]

    #remote use/download?
    #model_name = "sentence-transformers/all-MiniLM-L6-v2"
    #local use - use cli hf tool to download
    model_name = modelsPaths[0]
    model_kwargs = {"device": "cpu","trust_remote_code":True}
    encode_kwargs = {"normalize_embeddings": False}
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )



    #passive use of model. CODE LOADS MODEL AND USES IT
    #splitter=text_splitter.SemanticChunker(embeddings=hf) 
    splitter=text_splitter.SemanticChunker(embeddings=hf,buffer_size=1, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=75)

    
    

    temp=splitter.split_text(fullText)
    print("number of chunks: "+str(len(temp)))
    print(temp[0])
    print()
    print(temp[1])
    print()
    print(temp[2])
    print()
    print(temp[3])
    print()
    print(temp[4])
    

    print("hi")



def splitText(splitterModel,text) -> List[str]:
    chunks=splitterModel.split_text(text)
    return chunks

#dont use
def intersection(chunk1,chunk2):
    index1=[chunk1]
    index2=[chunk2]
    ret=[]

    temp=0
    for i in range(len(chunk1)):
        index1[i]=len(chunk1[i])+temp
        temp=index1[i]

    for i in range(len(chunk2)):
        index2[i]=len(chunk2[i])+temp
        temp=index2[i]

    inx=0
    cloIndex=0

    miC=min(len(chunk1),len(chunk2))


    for i in range(miC):
        a=0
        b=0
        print(chunk1[i])
        if index1[i+a] <index2[i+b]- len(chunk2[i+b]):
            a=a+1
        elif index1[i+a] - len(chunk1[i+a])>index2[i+b]:
            b=b+1


        inter=(index1[i+a]+index2[i+b])/2
        mi=min(index1[i+a],index2[i+b])
        ma=max(index1[i+a],index2[i+b])

        if mi -inter< inter -ma:
            cloIndex=mi
        else:
            cloIndex=ma


        if cloIndex == index1:
            ret[i]=chunk1[i+a][inx:]
        else:
            ret[i]=chunk2[i+b][inx:]

        inx=cloIndex

    if miC==chunk1:
        ret= ret + chunk1[miC:]
    else:
        ret= ret + chunk2[miC:]

    return ret

    #if index1<index2:
    #    
    #    if index1-inter<inter-index2
    #        cloIndex=index1
    #        cloText=chunk1
    #        ret[0]=chunk1[][inx:]
    #        inx=index1
    #else:
    #    if index1-inter>inter-index2
    #        cloIndex=index1
    #        cloText=chunk1
    #        ret[0]=chunk2[][inx:]
    #        inx=index2




if __name__ == '__main__':
    

    fullText="You better not lose it! The money is in the bag! By programmers do you mean hobbyists/students or do you mean professionals on the job? Hobbyists/students have difficulty learning things. Maybe they didn't pay attention or care during a databases/SQL class or maybe they had to do a bug fix in a codebase for a homework assignment and couldn't figure out how to do it. Those previous two sentences describe things that happened when I was getting my Computer Science degree. Oh, also I had issues with teamwork and cooperation with others on a team project. Teamwork is very important in the real world, professional software developers usually work in teams. Professionals on the job have to learn things too just like students do but they also have other difficulties. Professional programmers usually work in teams on a codebase that was created before they joined the team. The codebase can be massive, maybe millions of lines of code. They have to make bug fixes in that code and add features to it, and in order for them to be able to do that they must be able to read and understand the code. Reading and understanding code that other people wrote is really hard. I personally have more difficulty doing that than writing my own code by myself from scratch. Each codebase has its own structure and architecture that you must learn. Hopefully there is a senior (like 5+ years of experience) or staff/principal (like 10+ years of experience) developer who can explain it to you and maybe show you some pre-recorded video or documentation detailing and explaining the codebase. A codebase for work can take four months to two years to learn, depending on the codebase. There is a lot of code in a codebase. So yeah, that's a big problem. You have to make bug fixes and add features in that massive, hard to understand mess of code. Most beginners and newcomers think the code is a mess that is impossible to understand when they start, and their natural inclination is to want to rewrite everything from scratch in a way that they fully understand. A full rewrite is usually a bad idea even though you may feel you want to do it. A codebase that looks like an impossible to understand mess to you may make perfect sense and be easy to navigate to the person who initially created it and built it out from the beginning. It is better if this person with experience teaches you the codebase. It takes time to learn a codebase, and often the person paying you doesn't want to spend the time training you. Usually the person paying you just wants to see results for their money ASAP. There is this expression adding more new people to a late software project makes it later. The new people need to learn the codebase and be trained by the old people, and doing so takes time away from the old people that they could be spending working on the codebase. Often the person paying and hiring wants to speed up development by hiring more people, but the new people need to learn and be trained and that takes time. It is better to hire more people than you need early and fire off the people who suck or are bad at learning then it is to try to frantically hire and train people later on. The fact that the people paying money or hiring don't want to train makes it hard for newcomers with no experience (i.e. freshers) to enter the job market and become junior developers, which is the title people get when they have like 0-2 years of experience (it's basically an on-the-job apprenticeship where your mentor is a senior developer). Newcomers fresh out of school take more time to learn the ropes than people who have been doing the same job for years (i.e. senior developers). Junior developers need more ramp-up time, and employers don't want to pay for ramp-up time. For this reason, usually the first one or two coding jobs are the hardest for a computer science person to get. After they have the work experience, it's usually a lot easier to get a job but employers give them less ramp-up time then when they were first starting out. So getting your first programming job is a big problem for hobbyists/students and then after that the ramp-up is a big problem. Some people are slow at ramp-up and employers are impatient and they just fire off the slow people. Eventually people who are very slow keep getting fired and end up having to find a different industry. I was very bad at ramp-up and ultimately ended up unemployed and then on government disability benefits. So yeah, I would say the most common problems are getting your first job and the ramp-up. Learning technical stuff like programming languages and frameworks is a problem for some people but me personally, I used to enjoy reading books on that stuff that I bought off of Amazon before bed so that wasn't my main problem. Like if I needed to learn C I would read The C Programming Language book before bed or if I needed to learn Node.js I would read a book on that before bed and in general that combined with some time Googling and reading documentation on the job was enough. Once you learn a few programming languages, learning a new one isnt a problem and once you learn a few web frameworks like Node.js , Spring Boot, Django, Ruby on Rails, and so forth then learning an additional one isn't a problem. Hobbyists/students have some difficulty learning their first framework but for people with experience learning a new one isn't a problem. As for solutions, I dunno. It's hard to predict how good someone will be at a job before they spend time trying to do the job. There is a minimum barrier to entry but above that it is hard to predict. These problems like teamwork and ramp-up are systemic things and there isn't an easy fix. Like maybe if people were more intelligent or had more talent ramp-up would be easier. Some people have personality disorders like Narcissistic Personality Disorder which make teamwork with them difficult. There isn't an easy fix for these things. Edit: Another person replied Managers who dont understand what they want, what they need, and what is not possible. At Amazon I was blessed with a good, organized manager with technical skills so I didn't have the problem of a manager who didn't know what is not technically possible, but if your manager can't read or write code and has never built a code project himself then you can run into this issue."


    modeln=["sentence-transformers/all-mpnet-base-v2","sentence-transformers/all-MiniLM-L6-v2"]
    path="./models/"
    downloadModel(modeln[0])
    downloadModel(modeln[1])
    mod1=loadModel(modelName=modeln[0],modelPath=path)
    mod2=loadModel(modelName=modeln[1],modelPath=path)

    #mod=getFixedChunker(350,' ')
    #chunks=mod.create_documents(fullText)
    chunks1=splitText(mod1,fullText)
    chunks2=splitText(mod2,fullText)


    print(chunks1[0])
    print()
    print(chunks2[0])
    print()
    print()
    print(chunks1[1])
    print()
    print(chunks2[1])
    print()
    print()    
    print(chunks1[2])
    print()
    print(chunks2[2])
    print()
    print()


    
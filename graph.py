


from typing import Annotated
from typing import List
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END


import chunker



chk = chunker.getModel("sentence-transformers/all-MiniLM-L6-v2")


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


    fullText="You better not lose it! The money is in the bag! By programmers do you mean hobbyists/students or do you mean professionals on the job? Hobbyists/students have difficulty learning things. Maybe they didn't pay attention or care during a databases/SQL class or maybe they had to do a bug fix in a codebase for a homework assignment and couldn't figure out how to do it. Those previous two sentences describe things that happened when I was getting my Computer Science degree. Oh, also I had issues with teamwork and cooperation with others on a team project. Teamwork is very important in the real world, professional software developers usually work in teams. Professionals on the job have to learn things too just like students do but they also have other difficulties. Professional programmers usually work in teams on a codebase that was created before they joined the team. The codebase can be massive, maybe millions of lines of code. They have to make bug fixes in that code and add features to it, and in order for them to be able to do that they must be able to read and understand the code. Reading and understanding code that other people wrote is really hard. I personally have more difficulty doing that than writing my own code by myself from scratch. Each codebase has its own structure and architecture that you must learn. Hopefully there is a senior (like 5+ years of experience) or staff/principal (like 10+ years of experience) developer who can explain it to you and maybe show you some pre-recorded video or documentation detailing and explaining the codebase. A codebase for work can take four months to two years to learn, depending on the codebase. There is a lot of code in a codebase. So yeah, that's a big problem. You have to make bug fixes and add features in that massive, hard to understand mess of code. Most beginners and newcomers think the code is a mess that is impossible to understand when they start, and their natural inclination is to want to rewrite everything from scratch in a way that they fully understand. A full rewrite is usually a bad idea even though you may feel you want to do it. A codebase that looks like an impossible to understand mess to you may make perfect sense and be easy to navigate to the person who initially created it and built it out from the beginning. It is better if this person with experience teaches you the codebase. It takes time to learn a codebase, and often the person paying you doesn't want to spend the time training you. Usually the person paying you just wants to see results for their money ASAP. There is this expression adding more new people to a late software project makes it later. The new people need to learn the codebase and be trained by the old people, and doing so takes time away from the old people that they could be spending working on the codebase. Often the person paying and hiring wants to speed up development by hiring more people, but the new people need to learn and be trained and that takes time. It is better to hire more people than you need early and fire off the people who suck or are bad at learning then it is to try to frantically hire and train people later on. The fact that the people paying money or hiring don't want to train makes it hard for newcomers with no experience (i.e. freshers) to enter the job market and become junior developers, which is the title people get when they have like 0-2 years of experience (it's basically an on-the-job apprenticeship where your mentor is a senior developer). Newcomers fresh out of school take more time to learn the ropes than people who have been doing the same job for years (i.e. senior developers). Junior developers need more ramp-up time, and employers don't want to pay for ramp-up time. For this reason, usually the first one or two coding jobs are the hardest for a computer science person to get. After they have the work experience, it's usually a lot easier to get a job but employers give them less ramp-up time then when they were first starting out. So getting your first programming job is a big problem for hobbyists/students and then after that the ramp-up is a big problem. Some people are slow at ramp-up and employers are impatient and they just fire off the slow people. Eventually people who are very slow keep getting fired and end up having to find a different industry. I was very bad at ramp-up and ultimately ended up unemployed and then on government disability benefits. So yeah, I would say the most common problems are getting your first job and the ramp-up. Learning technical stuff like programming languages and frameworks is a problem for some people but me personally, I used to enjoy reading books on that stuff that I bought off of Amazon before bed so that wasn't my main problem. Like if I needed to learn C I would read The C Programming Language book before bed or if I needed to learn Node.js I would read a book on that before bed and in general that combined with some time Googling and reading documentation on the job was enough. Once you learn a few programming languages, learning a new one isnt a problem and once you learn a few web frameworks like Node.js , Spring Boot, Django, Ruby on Rails, and so forth then learning an additional one isn't a problem. Hobbyists/students have some difficulty learning their first framework but for people with experience learning a new one isn't a problem. As for solutions, I dunno. It's hard to predict how good someone will be at a job before they spend time trying to do the job. There is a minimum barrier to entry but above that it is hard to predict. These problems like teamwork and ramp-up are systemic things and there isn't an easy fix. Like maybe if people were more intelligent or had more talent ramp-up would be easier. Some people have personality disorders like Narcissistic Personality Disorder which make teamwork with them difficult. There isn't an easy fix for these things. Edit: Another person replied Managers who dont understand what they want, what they need, and what is not possible. At Amazon I was blessed with a good, organized manager with technical skills so I didn't have the problem of a manager who didn't know what is not technically possible, but if your manager can't read or write code and has never built a code project himself then you can run into this issue."

    temp=graph.invoke({"text": fullText})

    print(temp["chunks"][0])

    #for event in graph.stream({"text": fullText,"chunks": ["empty"],"monkeysPaw": 0}):
    #        for value in event.values():
    #            print("Assistant:", value["chunks"][0])





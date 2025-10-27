import pandas as pd
from lxml import etree
import tokenizers
import json
import tqdm
import pandas
import numpy
import chunker




def chunkLimitDocLimit(MaxChunks,MaxDocuments,totalchunks,totalDocuments) -> bool:
    if(MaxChunks <= totalchunks or MaxDocuments <= totalDocuments):
        return True
    else:
        return False



def parseXMLSection(MaxChunks,MaxDocuments,chunkSize):

    dataframe=pd.read_csv("/home/luke-weiss/dev/Biomedical-RAG-Agent/pmc_chunker/out/manifest_4000.csv")


    totalchunks=0
    totalDocuments=0

    dfs=[]
    for index,pmcId in enumerate(dataframe.iloc[:,0]):
    
        f = open("/home/luke-weiss/dev/Biomedical-RAG-Agent/pmc_chunker/data/xml/"+pmcId+".xml")
        xml=f.read()
        root = etree.fromstring(xml)


        abstract=root.find(".//abstract")
        body=root.find(".//body")

        if(abstract==None or body==None):
            continue


        chunkIndex=0
        
        for sec in abstract.xpath(".//sec"):
                
            #abstractText=""
            #for text in sec.xpath(".//p"):
            #    if( text.text ==None):
            #        continue
            #    abstractText=abstractText+text.text

            data=chunkSection(sec,dataframe,index,chunkIndex)
            dfs.append(pd.DataFrame(data))
            chunkIndex=chunkIndex+1
            totalchunks=totalchunks+1



        for sec in body.xpath(".//sec[not(ancestor::sec)]"):
        #for sec in body.xpath(".//sec[not(descendant::sec)]"):

            if len(sec.xpath(".//sec[not(descendant::sec)]")) == 0:

                data=chunkSection(sec,dataframe,index,chunkIndex)

                dfs.append(pd.DataFrame(data))
                chunkIndex=chunkIndex+1
                totalchunks=totalchunks+1

            for descendantSec in sec.xpath(".//sec[not(descendant::sec)]"):
                data=chunkSection(descendantSec,dataframe,index,chunkIndex)

                dfs.append(pd.DataFrame(data))
                chunkIndex=chunkIndex+1
                totalchunks=totalchunks+1
                if chunkLimitDocLimit(MaxChunks,MaxDocuments,len(dfs),totalDocuments):
                    break
        
            
            if chunkLimitDocLimit(MaxChunks,MaxDocuments,len(dfs),totalDocuments):
                break

        totalDocuments=totalDocuments+1
        
        if chunkLimitDocLimit(MaxChunks,MaxDocuments,totalchunks,totalDocuments):
            break
    
    #print(df)
    df = pd.concat(dfs, ignore_index=True)
    #print(dfs)
    df.to_json('./pmc_chunker/out/chunks.json', orient='records', lines=True)

def chunkSection(sec,dataframe,index,chunkIndex):
    
    sectionText=""
    for text in sec.xpath(".//p"):
        if text.text==None:
            continue
        sectionText=sectionText+text.text

    title=sec.xpath((".//title"))
    titSec=None
    if len(title)!=0:
        titSec=title[0].text
    #print(dataframe.loc[index,["topic"]])

    data={
        "PMCID":[dataframe.loc[index,"PMCID"]],
        "PMID":[dataframe.loc[index,"PMID"]],
        "doi":[dataframe.loc[index,"doi"]],
        "journal":[dataframe.loc[index,"journal"]],
        "year":[dataframe.loc[index,"year"]],
        "topic":[dataframe.loc[index,"topic"]],
        "article-title": [dataframe.loc[index,"title"]],
        "section_type": [sec.attrib.get('sec-type')],
        "section_id": [sec.attrib.get('id')],
        "section_title": [titSec],
        "section_path": [None],
        "chunk_text":[sectionText],
        "section_chunk_id": [0],
        "chunk_index": [chunkIndex], 
        "token_count": [len(sectionText)]
    }
    
    #print(chunkIndex)
    #chunkIndex=chunkIndex+1
    #totalchunks=totalchunks+1
    #df=pd.concat([df,pd.DataFrame(data)])
    return data

    #good explaination




def parseXMLSectionParagraph(MaxChunks,MaxDocuments,minchunkSize):

    dataframe=pd.read_csv("/home/luke-weiss/dev/Biomedical-RAG-Agent/pmc_chunker/out/manifest_4000.csv")


    totalchunks=0
    totalDocuments=0

    dfs=[]
    for index,pmcId in enumerate(dataframe.iloc[:,0]):
    
        f = open("/home/luke-weiss/dev/Biomedical-RAG-Agent/pmc_chunker/data/xml/"+pmcId+".xml")
        xml=f.read()
        root = etree.fromstring(xml)


        abstract=root.find(".//abstract")
        body=root.find(".//body")

        if(abstract==None or body==None):
            continue


        chunkIndex=0
        
        for sec in abstract.xpath(".//sec"):
                
            #abstractText=""
            #for text in sec.xpath(".//p"):
            #    if( text.text ==None):
            #        continue
            #    abstractText=abstractText+text.text

            data=chunkSection(sec,dataframe,index,chunkIndex)
            dfs.append(pd.DataFrame(data))
            chunkIndex=chunkIndex+1
            totalchunks=totalchunks+1



        for sec in body.xpath(".//sec[not(ancestor::sec)]"):
        #for sec in body.xpath(".//sec[not(descendant::sec)]"):

            if len(sec.xpath(".//sec[not(descendant::sec)]")) == 0:

                data=chunkSectionToParagraph(sec,dataframe,index,chunkIndex,minchunkSize)
                dfs.extend(data)
                chunkIndex=chunkIndex+len(data)
                totalchunks=totalchunks+len(data)
                
                if chunkLimitDocLimit(MaxChunks,MaxDocuments,len(dfs),totalDocuments):
                    break

            for descendantSec in sec.xpath(".//sec[not(descendant::sec)]"):
                data=chunkSectionToParagraph(descendantSec,dataframe,index,chunkIndex,minchunkSize)
                dfs.extend(data)
                chunkIndex=chunkIndex+len(data)
                totalchunks=totalchunks+len(data)
                
                if chunkLimitDocLimit(MaxChunks,MaxDocuments,len(dfs),totalDocuments):
                    break
        
            
            if chunkLimitDocLimit(MaxChunks,MaxDocuments,len(dfs),totalDocuments):
                break

        totalDocuments=totalDocuments+1
        
        if chunkLimitDocLimit(MaxChunks,MaxDocuments,totalchunks,totalDocuments):
            break
    
    #print(df)
    df = pd.concat(dfs, ignore_index=True)
    #print(dfs)
    df.to_json('./pmc_chunker/out/chunks.json', orient='records', lines=True)

def chunkSectionToParagraph(sec,dataframe,index,chunkIndex,minchunkSize):
    #https://medium.com/@larry.prestosa/speed-improvement-in-pandas-loop-df111f3f45ed

    dfs=[]

    title=sec.xpath((".//title"))
    titSec=None
    if len(title)!=0:
        titSec=title[0].text
    #print(dataframe.loc[index,["topic"]])


    #for minimum token count and too small sections. aka if section is too small then it wont be chunked. with this all sections get chunked even if it has little text
    numParagraphs=len(sec.xpath(".//p"))
    indexOfSectionChunk=0

    inputText=""
    for chunkText in sec.xpath(".//p"):


        if chunkText.text==None:
            indexOfSectionChunk=indexOfSectionChunk+1
            continue

        #if paragrpah is too small add it to the next chunk/ paragrph. if there is no other parapgraph IN THE SECTION then put it in it's own cluster 
        inputText=inputText+chunkText.text
        if len(inputText)<minchunkSize and indexOfSectionChunk+1 < numParagraphs:
            indexOfSectionChunk=indexOfSectionChunk+1
            continue


        data={
            "PMCID":[dataframe.loc[index,"PMCID"]],
            "PMID":[dataframe.loc[index,"PMID"]],
            "doi":[dataframe.loc[index,"doi"]],
            "journal":[dataframe.loc[index,"journal"]],
            "year":[dataframe.loc[index,"year"]],
            "topic":[dataframe.loc[index,"topic"]],
            "article-title": [dataframe.loc[index,"title"]],
            "section_type": [sec.attrib.get('sec-type')],
            "section_id": [str(sec.attrib.get('id'))],
            "section_title": [titSec],
            "section_path": [None],
            "chunk_text":[inputText],
            "section_chunk_id": [indexOfSectionChunk],
            "chunk_index": [chunkIndex], 
            "token_count": [len(inputText)]
        }
    
        #print(chunkIndex)
        chunkIndex=chunkIndex+1
        indexOfSectionChunk=indexOfSectionChunk+1
        temp=pd.DataFrame(data)

        dfs.append(temp)


        inputText=""
        #print(temp.shape)
        #df=pd.concat([df,temp])
    #print(len(dfs))

    return dfs








def parseXMLSectionParagraphModel(MaxChunks,MaxDocuments,minchunkSize,chunkingModel):

    dataframe=pd.read_csv("/home/luke-weiss/dev/Biomedical-RAG-Agent/pmc_chunker/out/manifest_4000.csv")


    totalchunks=0
    totalDocuments=0

    dfs=[]
    for index,pmcId in enumerate(dataframe.iloc[:,0]):
    
        f = open("/home/luke-weiss/dev/Biomedical-RAG-Agent/pmc_chunker/data/xml/"+pmcId+".xml")
        xml=f.read()
        root = etree.fromstring(xml)


        abstract=root.find(".//abstract")
        body=root.find(".//body")

        if(abstract==None or body==None):
            continue


        chunkIndex=0
        
        for sec in abstract.xpath(".//sec"):
                
            #abstractText=""
            #for text in sec.xpath(".//p"):
            #    if( text.text ==None):
            #        continue
            #    abstractText=abstractText+text.text

            data=chunkSection(sec,dataframe,index,chunkIndex)
            dfs.append(pd.DataFrame(data))
            chunkIndex=chunkIndex+1
            totalchunks=totalchunks+1



        for sec in body.xpath(".//sec[not(ancestor::sec)]"):
        #for sec in body.xpath(".//sec[not(descendant::sec)]"):

            if len(sec.xpath(".//sec[not(descendant::sec)]")) == 0:

                data=chunkSectionParagraphwModel(sec,dataframe,index,chunkIndex,minchunkSize,chunkingModel)
                dfs.extend(data)
                chunkIndex=chunkIndex+len(data)
                totalchunks=totalchunks+len(data)
                
                if chunkLimitDocLimit(MaxChunks,MaxDocuments,len(dfs),totalDocuments):
                    break

            for descendantSec in sec.xpath(".//sec[not(descendant::sec)]"):
                data=chunkSectionParagraphwModel(descendantSec,dataframe,index,chunkIndex,minchunkSize,chunkingModel)
                dfs.extend(data)
                chunkIndex=chunkIndex+len(data)
                totalchunks=totalchunks+len(data)
                
                if chunkLimitDocLimit(MaxChunks,MaxDocuments,len(dfs),totalDocuments):
                    break
        
            
            if chunkLimitDocLimit(MaxChunks,MaxDocuments,len(dfs),totalDocuments):
                break

        totalDocuments=totalDocuments+1
        
        if chunkLimitDocLimit(MaxChunks,MaxDocuments,totalchunks,totalDocuments):
            break
    
    #print(df)
    df = pd.concat(dfs, ignore_index=True)
    #print(dfs)
    df.to_json('./pmc_chunker/out/chunks.json', orient='records', lines=True)

def chunkSectionParagraphwModel(sec,dataframe,index,chunkIndex,minchunkSize,chunkingModel):
    #https://medium.com/@larry.prestosa/speed-improvement-in-pandas-loop-df111f3f45ed

    dfs=[]

    title=sec.xpath((".//title"))
    titSec=None
    if len(title)!=0:
        titSec=title[0].text
    #print(dataframe.loc[index,["topic"]])


    #for minimum token count and too small sections. aka if section is too small then it wont be chunked. with this all sections get chunked even if it has little text
    numParagraphs=len(sec.xpath(".//p"))
    indexOfSectionChunk=0
    
    inputText=""
    for chunkText in sec.xpath(".//p"):


        if chunkText.text==None:
            indexOfSectionChunk=indexOfSectionChunk+1
            continue

        #if paragrpah is too small add it to the next chunk/ paragrph. if there is no other parapgraph IN THE SECTION then put it in it's own cluster 
        inputText=inputText+chunkText.text
        if len(inputText)<minchunkSize and indexOfSectionChunk+1 < numParagraphs:
            indexOfSectionChunk=indexOfSectionChunk+1
            continue

        
        for paragraphText in chunkingModel.split_text(inputText):

            data={
                "PMCID":[dataframe.loc[index,"PMCID"]],
                "PMID":[dataframe.loc[index,"PMID"]],
                "doi":[dataframe.loc[index,"doi"]],
                "journal":[dataframe.loc[index,"journal"]],
                "year":[dataframe.loc[index,"year"]],
                "topic":[dataframe.loc[index,"topic"]],
                "article-title": [dataframe.loc[index,"title"]],
                "section_type": [sec.attrib.get('sec-type')],
                "section_id": [str(sec.attrib.get('id'))],
                "section_title": [titSec],
                "section_path": [None],
                "chunk_text":[paragraphText],
                "section_chunk_id": [indexOfSectionChunk],
                "chunk_index": [chunkIndex], 
                "token_count": [len(paragraphText)]
            }
    
            #print(chunkIndex)
            chunkIndex=chunkIndex+1
            indexOfSectionChunk=indexOfSectionChunk+1
            numParagraphs=numParagraphs+1
            temp=pd.DataFrame(data)

            dfs.append(temp)


            inputText=""
        #print(temp.shape)
        #df=pd.concat([df,temp])
    #print(len(dfs))

    return dfs



def getBlankDataframe():
    return pd.DataFrame(columns=["article-title","PMCID","PMID","doi","journal","year","topic","section_type","section_id","section_title","section_path","chunk_text", "section_chunk_id","chunk_index","token_count"])

if __name__ == '__main__':
    #parseXMLSection(1000,4000,340)

    #parseXMLSectionParagraph(100000,4000,700)

    chk=chunker.getFixedChunker(700)
    #chk=chunker.getModel("sentence-transformers/all-MiniLM-L6-v2")
    parseXMLSectionParagraphModel(100000,15,700,chk)

import pandas as pd
from lxml import etree
import tokenizers
import json
import tqdm
import pandas
import numpy
import chunker

chk=chunker.getFixedChunker(350)


def chunkLimitDocLimit(MaxChunks,MaxDocuments,totalchunks,totalDocuments) -> bool:
    if(MaxChunks <= totalchunks or MaxDocuments <= totalDocuments):
        return True
    else:
        return False


#dont use. need as reference for later changes
def ORIGINALparseXMLSection(MaxChunks,MaxDocuments):

    dataframe=pd.read_csv("/home/luke-weiss/dev/Biomedical-RAG-Agent/pmc_chunker/out/manifest_4000.csv")

    

    


    totalchunks=0
    totalDocuments=0

    df=pd.DataFrame(columns=["article-title","PMCID","PMID","doi","journal","year","topic","section_type","section_id","section_title","chunk_text","section_path","chunk_index","token_count"])
    for index,pmcId in enumerate(dataframe.iloc[:,0]):
    
        f = open("/home/luke-weiss/dev/Biomedical-RAG-Agent/pmc_chunker/data/xml/"+pmcId+".xml")
        xml=f.read()


        root = etree.fromstring(xml)

        articalTitle=root.find(".//article-title").text
        abstract=root.find(".//abstract")

        body=root.find(".//body")

        if(abstract==None or body==None):
            continue



        chunkIndex=0
        for sec in abstract.xpath(".//sec"):
            title=sec.xpath((".//title"))
            if len(title)==0:
                continue
            titSec=title[0].text
            abstractText=""
            for text in sec.xpath(".//p"):
                if(abstractText==None or text.text ==None):
                    continue
                abstractText=abstractText+text.text

            #print(dataframe.loc[index,["topic"]])
            data={
                "PMCID":[dataframe.loc[index,"PMCID"]],
                "PMID":[dataframe.loc[index,"PMID"]],
                "doi":[dataframe.loc[index,"doi"]],
                "journal":[dataframe.loc[index,"journal"]],
                "year":[dataframe.loc[index,"year"]],
                "topic":[dataframe.loc[index,"topic"]],
                "article-title": [articalTitle],
                "section_type": [sec.attrib.get('sec-type')],
                "section_id": [sec.attrib.get('id')],
                "section_title": [titSec],
                "chunk_text":[abstractText],
                "section_path": [None],
                "chunk_index": [chunkIndex], 
                "token_count": [len(abstractText)]
            }
            chunkIndex=chunkIndex+1
            totalchunks=totalchunks+1
            df=pd.concat([df,pd.DataFrame(data)])








        for sec in body.xpath(".//sec"):
            title=sec.xpath((".//title"))
            if len(title)==0:
                continue
            titSec=title[0].text
            sectionText=""
            for text in sec.xpath(".//p"):
                if text.text==None:
                    continue
                sectionText=sectionText+text.text

            #print(dataframe.loc[index,["topic"]])
            data={
                "PMCID":[dataframe.loc[index,"PMCID"]],
                "PMID":[dataframe.loc[index,"PMID"]],
                "doi":[dataframe.loc[index,"doi"]],
                "journal":[dataframe.loc[index,"journal"]],
                "year":[dataframe.loc[index,"year"]],
                "topic":[dataframe.loc[index,"topic"]],
                "article-title": [articalTitle],
                "section_type": [sec.attrib.get('sec-type')],
                "section_id": [sec.attrib.get('id')],
                "section_title": [titSec],
                "chunk_text":[sectionText],
                "section_path": [None],
                "chunk_index": [chunkIndex], 
                "token_count": [len(sectionText)]
            }
            
            #print(chunkIndex)
            chunkIndex=chunkIndex+1
            totalchunks=totalchunks+1
            df=pd.concat([df,pd.DataFrame(data)])
            #df[df.shape[1]]=pd.DataFrame(data)

        
            if chunkLimitDocLimit(MaxChunks,MaxDocuments,totalchunks,totalDocuments):
                break

        totalDocuments=totalDocuments+1

        if chunkLimitDocLimit(MaxChunks,MaxDocuments,totalchunks,totalDocuments):
            break

    #print(df)
    df.to_json('./pmc_chunker/out/chunks.json', orient='records', lines=True)

    
    #bodText=body.xpath(".//p")
    #print(bodText[0].text)


    #figures=root.xpath(".//fig")
    #print(figures)



def parseXMLSection(MaxChunks,MaxDocuments,chunkSize):

    dataframe=pd.read_csv("/home/luke-weiss/dev/Biomedical-RAG-Agent/pmc_chunker/out/manifest_4000.csv")


    totalchunks=0
    totalDocuments=0

    df=pd.DataFrame(columns=["article-title","PMCID","PMID","doi","journal","year","topic","section_type","section_id","section_title","chunk_text","section_path","chunk_index","token_count"])
    for index,pmcId in enumerate(dataframe.iloc[:,0]):
    
        f = open("/home/luke-weiss/dev/Biomedical-RAG-Agent/pmc_chunker/data/xml/"+pmcId+".xml")
        xml=f.read()


        root = etree.fromstring(xml)

        articalTitle=root.find(".//article-title").text
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

            df=pd.concat([df,convertSectionToDataframe(sec,dataframe,index,chunkIndex)])
            chunkIndex=chunkIndex+1
            totalchunks=totalchunks+1




        chunksFrame=None
        for sec in body.xpath(".//sec[not(ancestor::sec)]"):
        #for sec in body.xpath(".//sec[not(descendant::sec)]"):


            sectionText=""
            for text in sec.xpath(".//p"):
                if text.text==None:
                    continue
                sectionText=sectionText+text.text

            if len(sectionText) > chunkSize:
                chunksFrame=breakDownSection(sec,dataframe,index,chunkIndex)
                if not chunksFrame.empty:
                    df=pd.concat([df,chunksFrame])
                    chunkIndex=chunkIndex+chunksFrame.shape[0]
                    
                    

                else:
                    #HERE DO MANUAL CHUNKING. IT CANT BE SEPARATED ENOUGH
                    # temp=convertSectionToDataframe(sec,dataframe,index,chunkIndex)
                    # df=pd.concat([df,temp])
                    # chunkIndex=chunkIndex+1

                    chunksFrame=chunkSingularSection(sec,dataframe,index,chunkIndex)
                    df=pd.concat([df,chunksFrame])
                    chunkIndex=chunkIndex+chunksFrame.shape[0]



            else:
                df=pd.concat([df,convertSectionToDataframe(sec,dataframe,index,chunkIndex)])
                chunkIndex=chunkIndex+1

        
            
            if chunkLimitDocLimit(MaxChunks,MaxDocuments,df.shape[0],totalDocuments):
                break

        totalDocuments=totalDocuments+1
        
        if chunkLimitDocLimit(MaxChunks,MaxDocuments,totalchunks,totalDocuments):
            break
    
    #print(df)
    

    df.to_json('./pmc_chunker/out/chunks.json', orient='records', lines=True)



 

def breakDownSection(section,dataframe,index,chunkIndex):

    dfs=[getBlankDataframe()]
    ret=None
    #for descendantSec in section.xpath(".//sec[not(ancestor::sec)]"):
    for descendantSec in section.xpath(".//sec[not(descendant::sec)]"):
        frame=chunkSingularSection(descendantSec,dataframe,index,chunkIndex)
        if not frame.empty:
            dfs.append(frame)
            #ret=pd.concat([ret,frame],ignore_index=True)
            chunkIndex=chunkIndex+1
        else:
            continue
        

        # frame=convertSectionToDataframe(descendantSec,dataframe,index,chunkIndex)
        # if not frame.empty:
        #     dfs.append(frame)
        #     chunkIndex=chunkIndex+1
        # else:
        #     continue

    ret=pd.concat(dfs, ignore_index=True)
    return ret

def convertSectionToDataframe(sec,dataframe,index,chunkIndex):
    
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
        "chunk_text":[sectionText],
        "section_path": [None],
        "chunk_index": [chunkIndex], 
        "token_count": [len(sectionText)]
    }
    
    #print(chunkIndex)
    #chunkIndex=chunkIndex+1
    #totalchunks=totalchunks+1
    #df=pd.concat([df,pd.DataFrame(data)])
    return pd.DataFrame(data)

def chunkSingularSection(sec,dataframe,index,chunkIndex):
    #good explaination
    #https://medium.com/@larry.prestosa/speed-improvement-in-pandas-loop-df111f3f45ed

    dfs=[]
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

    for indexOfSectionChunk, chunkText in enumerate(chk.split_text(sectionText)):

        data={
            "PMCID":[dataframe.loc[index,"PMCID"]],
            "PMID":[dataframe.loc[index,"PMID"]],
            "doi":[dataframe.loc[index,"doi"]],
            "journal":[dataframe.loc[index,"journal"]],
            "year":[dataframe.loc[index,"year"]],
            "topic":[dataframe.loc[index,"topic"]],
            "article-title": [dataframe.loc[index,"title"]],
            "section_type": [sec.attrib.get('sec-type')],
            "section_id": [str(sec.attrib.get('id'))+"-"+str(indexOfSectionChunk)],
            "section_title": [titSec],
            "chunk_text":[chunkText],
            "section_path": [None],
            "chunk_index": [chunkIndex], 
            "token_count": [len(chunkText)]
        }
    
        #print(chunkIndex)
        chunkIndex=chunkIndex+1
        temp=pd.DataFrame(data)

        dfs.append(temp)
        #print(temp.shape)
        #df=pd.concat([df,temp])
    #print(len(dfs))
    if len(dfs)!=0:
        ret=pd.concat(dfs, ignore_index=True)
    else:
        ret=getBlankDataframe()
    return ret

def getBlankDataframe():
    return pd.DataFrame(columns=["article-title","PMCID","PMID","doi","journal","year","topic","section_type","section_id","section_title","chunk_text","section_path","chunk_index","token_count"])

if __name__ == '__main__':
    parseXMLSection(10000,4000,340)

    #ORIGINALparseXMLSection(36000,2)
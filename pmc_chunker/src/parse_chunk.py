import pandas as pd
from lxml import etree
import tokenizers
import json
import tqdm
import pandas
import numpy


def parseXML(MaxChunks,MaxDocuments):

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


        
        totalDocuments=totalDocuments+1
        if(MaxChunks < totalchunks or MaxDocuments < totalDocuments):
            break


    df.to_json('./../out/chunks.json', orient='records', lines=True)

    
    #bodText=body.xpath(".//p")
    #print(bodText[0].text)


    #figures=root.xpath(".//fig")
    #print(figures)


if __name__ == '__main__':
    parseXML(36000,4000)
import pandas as pd
from lxml import etree
import chunker
import os


def getDataframe(path,xmlFolder=False):
    
    names=[]
    frame=None

    if xmlFolder:
        for filename in os.listdir(path):
            if not filename.endswith('.xml'): continue
            names.append(os.path.splitext(filename)[0])
        

        frame = pd.DataFrame(names, columns=['PMCID'])
    else:
        frame=pd.read_csv(path)
    return frame

def getXML(name,path="./pmc_chunker/data/xml/"):
    f=open(path+name)
    return f

def chunkLimitDocLimit(MaxChunks,MaxDocuments,totalchunks,totalDocuments) -> bool:
    if(MaxChunks <= totalchunks or MaxDocuments <= totalDocuments):
        return True
    else:
        return False

def getMeta(root,dataframe,index):
    metadata={
        "PMID": int(root.xpath("//article-meta/article-id[@pub-id-type=\"pmid\"]/text()")[0]),
        "PMCID":"PMC" +root.xpath("//article-meta/article-id[@pub-id-type=\"pmc\"]/text()")[0],
        #"doi":root.xpath(".//article-meta/article-id[@pub-id-type=\"doi\"]")[0].text,
        "doi":root.findtext(".//article-meta/article-id[@pub-id-type=\"doi\"]"),
        "journal":root.xpath("//article//journal-title/text()")[0],
        "year":int(root.xpath("//article//year/text()")[0]),
        "article-title":"".join(root.xpath("//article-meta//article-title//text()")),
        "topic":None
    }

    if "topic" in dataframe:
        metadata["topic"]=dataframe["topic"][index]


    # metadata={
    #     "PMCID":dataframe.PMCID.get(index,None),
    #     "PMID":dataframe.PMID.get(index,None),
    #     "doi":dataframe.doi.get(index,None),
    #     "journal":dataframe.journal.get(index,None),
    #     "year":dataframe.year.get(index,None),
    #     "topic":dataframe.topic.get(index,None),
    #     "article-title":dataframe.title.get(index,None),
    # }


    return metadata



def parseXMLSection(sourceFileFrame,MaxChunks,MaxDocuments):


    totalchunks=0
    totalDocuments=0

    dfs=[]
    for index,pmcId in enumerate(sourceFileFrame["PMCID"]):
        #f = open("./pmc_chunker/data/xml/"+pmcId+".xml")
        f = getXML(pmcId+".xml")
        
        xml=f.read()
        root = etree.fromstring(xml)


        abstract=root.find(".//abstract")
        body=root.find(".//body")

        if(abstract==None or body==None):
            continue



        metadata=getMeta(root,sourceFileFrame,index)

        chunkIndex=0
        
        for sec in abstract.xpath(".//sec"):
                
            #abstractText=""
            #for text in sec.xpath(".//p"):
            #    if( text.text ==None):
            #        continue
            #    abstractText=abstractText+text.text

            data=chunkSection(sec,metadata,chunkIndex)
            dfs.append(pd.DataFrame(data))
            chunkIndex=chunkIndex+1
            totalchunks=totalchunks+1



        for sec in body.xpath(".//sec[not(ancestor::sec)]"):
        #for sec in body.xpath(".//sec[not(descendant::sec)]"):

            if len(sec.xpath(".//sec[not(descendant::sec)]")) == 0:

                data=chunkSection(sec,metadata,chunkIndex)

                dfs.append(pd.DataFrame(data))
                chunkIndex=chunkIndex+1
                totalchunks=totalchunks+1

            for descendantSec in sec.xpath(".//sec[not(descendant::sec)]"):
                data=chunkSection(descendantSec,metadata,chunkIndex)

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

def chunkSection(sec,metadata,chunkIndex):
    
    sectionText=""
    for text in sec.xpath(".//p"):
        #if text.text==None:
        #   continue
        #sectionText=sectionText+text.text
        sectionText=sectionText+"".join(text.xpath(".//text()"))
        
    #title=sec.xpath((".//title"))
    title=sec.xpath(("title"))
    titSec=None
    if len(title)!=0:
        titSec=title[0].text
    #print(dataframe.loc[index,["topic"]])

    data={
        "PMCID":[metadata["PMCID"]],
        "PMID":[metadata["PMID"]],
        "doi":[metadata["doi"]],
        "journal":[metadata["journal"]],
        "year":[metadata["year"]],
        "topic":[metadata["topic"]],
        "article-title": [metadata["article-title"]],
        "chunk_type":["text"],#[sec.attrib.get('sec-type')],
        "section_id": [sec.attrib.get('id')],
        "section_title": [titSec],
        "sub_section_title": [None],
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




def parseXMLSectionParagraph(sourceFileFrame,MaxChunks,MaxDocuments,minchunkSize):


    totalchunks=0
    totalDocuments=0

    dfs=[]



    for index,pmcId in enumerate(sourceFileFrame["PMCID"]):
    
        #f = open("./pmc_chunker/data/xml/"+pmcId+".xml")
        f = getXML(pmcId+".xml")
        
        xml=f.read()
        root = etree.fromstring(xml)


        abstract=root.find(".//abstract")
        body=root.find(".//body")

        if(abstract==None or body==None):
            continue


        metadata=getMeta(root,sourceFileFrame,index)



        chunkIndex=0
        
        # for sec in abstract.xpath(".//sec"):
        # data=chunkSection(sec,metadata,chunkIndex)
        data=chunkSection(abstract,metadata,chunkIndex)
        data=pd.DataFrame(data)
        if data.loc[0,"section_title"]==None:
            data.loc[0,"section_title"]="abstract"
        dfs.append(data)
        chunkIndex=chunkIndex+1
        totalchunks=totalchunks+1


        figs=chunkFigure(root,None,None,metadata,chunkIndex)
        dfs.extend(figs)
        chunkIndex=chunkIndex+len(figs)
        totalchunks=totalchunks+len(figs)



        for sec in body.xpath(".//sec[not(ancestor::sec)]"):
        #for sec in body.xpath(".//sec[not(descendant::sec)]"):


            sectionTitle=None
            if len(sec.xpath("title")) != 0:
                sectionTitle=sec.xpath("title")[0].text


            subSections=sec.xpath(".//sec[not(descendant::sec)]")

            if len(subSections) == 0:


                 #section text
                data=chunkSectionParagraph(sec,sectionTitle,None,metadata,chunkIndex,minchunkSize)
                dfs.extend(data)
                chunkIndex=chunkIndex+len(data)
                totalchunks=totalchunks+len(data)


                #section figures
                #print("here1")
                # figs=chunkFigure(sec,sectionTitle,None,metadata,chunkIndex)
                # dfs.extend(figs)
                # chunkIndex=chunkIndex+len(figs)
                # totalchunks=totalchunks+len(figs)

                if chunkLimitDocLimit(MaxChunks,MaxDocuments,len(dfs),totalDocuments):
                    break

            for sub in subSections:
                subTitle=None
                if len(sub.xpath("title")) != 0:
                    subTitle=sub.xpath("title")[0].text


                #sub section text
                data=chunkSubSectionParagraph(sub,sectionTitle,subTitle,metadata,chunkIndex,minchunkSize)
                dfs.extend(data)
                chunkIndex=chunkIndex+len(data)
                totalchunks=totalchunks+len(data)
                

                #sub section figures
                #print("here2")
                # figs=chunkFigure(sub,sectionTitle,subTitle,metadata,chunkIndex)
                # dfs.extend(figs)
                # chunkIndex=chunkIndex+len(figs)
                # totalchunks=totalchunks+len(figs)


                if chunkLimitDocLimit(MaxChunks,MaxDocuments,len(dfs),totalDocuments):
                    break
        
        
            # figs=chunkFigure(sec,sectionTitle,None,metadata,chunkIndex)
            # dfs.extend(figs)
            # chunkIndex=chunkIndex+len(figs)
            # totalchunks=totalchunks+len(figs)

            if chunkLimitDocLimit(MaxChunks,MaxDocuments,len(dfs),totalDocuments):
                break

        totalDocuments=totalDocuments+1
        
        if chunkLimitDocLimit(MaxChunks,MaxDocuments,totalchunks,totalDocuments):
            break
    
    #print(df)
    df = pd.concat(dfs, ignore_index=True)
    #print(dfs)
    df.to_json('./pmc_chunker/out/chunks.json', orient='records', lines=True)

def chunkSectionParagraph(sec,sectionTitle,subSectionTitle,metadata,chunkIndex,minchunkSize):
    #https://medium.com/@larry.prestosa/speed-improvement-in-pandas-loop-df111f3f45ed

    dfs=[]


    #for minimum token count and too small sections. aka if section is too small then it wont be chunked. with this all sections get chunked even if it has little text
    numParagraphs=len(sec.xpath(".//p"))
    indexOfSectionChunk=0

    inputText=""
    for chunkText in sec.xpath(".//p"):


        #if chunkText.text==None:
        # if len(chunkText.xpath(".//text()"))==0:
        #     indexOfSectionChunk=indexOfSectionChunk+1
        #     continue

        #if paragrpah is too small add it to the next chunk/ paragrph. if there is no other parapgraph IN THE SECTION then put it in it's own cluster 
        #inputText=inputText+chunkText.text
        inputText=inputText+"".join(chunkText.xpath(".//text()"))

        if len(inputText)<minchunkSize and indexOfSectionChunk+1 < numParagraphs:
            indexOfSectionChunk=indexOfSectionChunk+1
            continue


        data={
            "PMCID":[metadata["PMCID"]],
            "PMID":[metadata["PMID"]],
            "doi":[metadata["doi"]],
            "journal":[metadata["journal"]],
            "year":[metadata["year"]],
            "topic":[metadata["topic"]],
            "article-title": [metadata["article-title"]],
            "chunk_type":["text"],#[sec.attrib.get('sec-type')],
            "section_id": [sec.attrib.get('id')],
            "section_title": [sectionTitle],
            "sub_section_title": [subSectionTitle],
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

def chunkSubSectionParagraph(sec,sectionTitle,subSectionTitle,metadata,chunkIndex,minchunkSize):
    #https://medium.com/@larry.prestosa/speed-improvement-in-pandas-loop-df111f3f45ed

    dfs=[]


    #for minimum token count and too small sections. aka if section is too small then it wont be chunked. with this all sections get chunked even if it has little text
    numParagraphs=len(sec.xpath(".//p"))
    indexOfSectionChunk=0

    inputText=""
    for chunkText in sec.xpath(".//p"):


        #if chunkText.text==None:
        # if len(chunkText.xpath(".//text()"))==0:
        #     indexOfSectionChunk=indexOfSectionChunk+1
        #     continue

        #if paragrpah is too small add it to the next chunk/ paragrph. if there is no other parapgraph IN THE SECTION then put it in it's own cluster 
        #inputText=inputText+chunkText.text
        inputText=inputText+"".join(chunkText.xpath(".//text()"))

        if len(inputText)<minchunkSize and indexOfSectionChunk+1 < numParagraphs:
            indexOfSectionChunk=indexOfSectionChunk+1
            continue


        data={
            "PMCID":[metadata["PMCID"]],
            "PMID":[metadata["PMID"]],
            "doi":[metadata["doi"]],
            "journal":[metadata["journal"]],
            "year":[metadata["year"]],
            "topic":[metadata["topic"]],
            "article-title": [metadata["article-title"]],
            "chunk_type":["text"],#[sec.attrib.get('sec-type')],
            "section_id": [sec.attrib.get('id')],
            "section_title": [sectionTitle],
            "sub_section_title": [subSectionTitle],
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



def parseXMLSectionParagraphModel(sourceFileFrame,MaxChunks,MaxDocuments,minchunkSize,chunkingModel):

    

    totalchunks=0
    totalDocuments=0


    dfs=[]
    for index,pmcId in enumerate(sourceFileFrame["PMCID"]):
    
        #f = open("./pmc_chunker/data/xml/"+pmcId+".xml")
        f = getXML(pmcId+".xml")
        
        xml=f.read()
        root = etree.fromstring(xml)


        abstract=root.find(".//abstract")
        body=root.find(".//body")

        if(abstract==None or body==None):
            continue


        
        metadata=getMeta(root,sourceFileFrame,index)

 





        chunkIndex=0
        

        data=chunkSection(abstract,metadata,chunkIndex)
        temp=pd.DataFrame(data)
        if temp.loc[0,"section_title"]==None:
            temp.loc[0,"section_title"]="abstract"
        dfs.append(temp)
        chunkIndex=chunkIndex+1
        totalchunks=totalchunks+1



        for sec in body.xpath(".//sec[not(ancestor::sec)]"):
        #for sec in body.xpath(".//sec[not(descendant::sec)]"):

            sectionTitle=None
            if len(sec.xpath("title")) != 0:
                sectionTitle=sec.xpath("title")[0].text


            subSections=sec.xpath(".//sec[not(descendant::sec)]")


            if len(subSections) == 0:

                #section text
                data=chunkSectionParagraphwModel(sec,sectionTitle,None,metadata,chunkIndex,minchunkSize,chunkingModel)
                dfs.extend(data)
                chunkIndex=chunkIndex+len(data)
                totalchunks=totalchunks+len(data)
                


                #section figures
                #print("here1")
                # figs=chunkFigure(sec,sectionTitle,None,metadata,chunkIndex)
                # dfs.extend(figs)
                # chunkIndex=chunkIndex+len(figs)
                # totalchunks=totalchunks+len(figs)


                if chunkLimitDocLimit(MaxChunks,MaxDocuments,len(dfs),totalDocuments):
                    break

            for sub in subSections:
                
                subTitle=None
                if len(sub.xpath("title")) != 0:
                    subTitle=sub.xpath("title")[0].text
                

                #sub section text
                data=chunkSubSectionParagraphwModel(sub,sectionTitle,subTitle,metadata,chunkIndex,minchunkSize,chunkingModel)
                dfs.extend(data)
                chunkIndex=chunkIndex+len(data)
                totalchunks=totalchunks+len(data)
                


                #sub section figures
                #print("here2")
                # figs=chunkFigure(sub,sectionTitle,subTitle,metadata,chunkIndex)
                # dfs.extend(figs)
                # chunkIndex=chunkIndex+len(figs)
                # totalchunks=totalchunks+len(figs)


                if chunkLimitDocLimit(MaxChunks,MaxDocuments,len(dfs),totalDocuments):
                    break
        
            figs=chunkFigure(sec,sectionTitle,None,metadata,chunkIndex)
            dfs.extend(figs)
            chunkIndex=chunkIndex+len(figs)
            totalchunks=totalchunks+len(figs)
            
            if chunkLimitDocLimit(MaxChunks,MaxDocuments,len(dfs),totalDocuments):
                break

        totalDocuments=totalDocuments+1
        
        if chunkLimitDocLimit(MaxChunks,MaxDocuments,totalchunks,totalDocuments):
            break
    
    #print(df)
    df = pd.concat(dfs, ignore_index=True)
    #print(dfs)
    #print(len(df))
    #print(len(df[df["token_count"]<300]))
    #print(len(df[df["token_count"]<300])/len(df))
    #print(len(df[df["token_count"]<400])/len(df))
    #print(len(df[df["token_count"]<500])/len(df))
    #print(len(df[df["token_count"]<600])/len(df))
    df.to_json('./pmc_chunker/out/chunks.json', orient='records', lines=True)

def chunkSectionParagraphwModel(sec,sectionTitle,subSectionTitle,metadata,chunkIndex,minchunkSize,chunkingModel):
    #https://medium.com/@larry.prestosa/speed-improvement-in-pandas-loop-df111f3f45ed

    dfs=[]



    #for minimum token count and too small sections. aka if section is too small then it wont be chunked. with this all sections get chunked even if it has little text
    numParagraphs=len(sec.xpath(".//p"))
    indexOfSectionChunk=0
    
    inputText=""
    for chunkText in sec.xpath(".//p"):




        #if paragrpah is too small add it to the next chunk/ paragrph. if there is no other parapgraph IN THE SECTION then put it in it's own cluster 
        #inputText=inputText+chunkText.text
        inputText=inputText+"".join(chunkText.xpath(".//text()"))
        


        #if chunkText.text==None:
        # if len(inputText)==0:
        #    indexOfSectionChunk=indexOfSectionChunk+1
        #    continue

        if len(inputText)<minchunkSize and indexOfSectionChunk+1 < numParagraphs:
            indexOfSectionChunk=indexOfSectionChunk+1
            continue

        
        for paragraphText in chunkingModel.split_text(inputText):

            data={
                "PMCID":[metadata["PMCID"]],
                "PMID":[metadata["PMID"]],
                "doi":[metadata["doi"]],
                "journal":[metadata["journal"]],
                "year":[metadata["year"]],
                "topic":[metadata["topic"]],
                "article-title": [metadata["article-title"]],
                "chunk_type":["text"],#[sec.attrib.get('sec-type')],
                "section_id": [sec.attrib.get('id')],
                "section_title": [sectionTitle],
                "sub_section_title": [subSectionTitle],
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

def chunkSubSectionParagraphwModel(sec,sectionTitle,subSectionTitle,metadata,chunkIndex,minchunkSize,chunkingModel):
    #https://medium.com/@larry.prestosa/speed-improvement-in-pandas-loop-df111f3f45ed

    dfs=[]




    #for minimum token count and too small sections. aka if section is too small then it wont be chunked. with this all sections get chunked even if it has little text
    numParagraphs=len(sec.xpath(".//p"))
    indexOfSectionChunk=0
    
    inputText=""
    for chunkText in sec.xpath(".//p"):




        #if paragrpah is too small add it to the next chunk/ paragrph. if there is no other parapgraph IN THE SECTION then put it in it's own cluster 
        #inputText=inputText+chunkText.text
        inputText=inputText+"".join(chunkText.xpath(".//text()"))
        


        #if chunkText.text==None:
        # if len(inputText)==0:
        #    indexOfSectionChunk=indexOfSectionChunk+1
        #    continue

        if len(inputText)<minchunkSize and indexOfSectionChunk+1 < numParagraphs:
            indexOfSectionChunk=indexOfSectionChunk+1
            continue

        
        for paragraphText in chunkingModel.split_text(inputText):

            data={
                "PMCID":[metadata["PMCID"]],
                "PMID":[metadata["PMID"]],
                "doi":[metadata["doi"]],
                "journal":[metadata["journal"]],
                "year":[metadata["year"]],
                "topic":[metadata["topic"]],
                "article-title": [metadata["article-title"]],
                "chunk_type":["text"],#[sec.attrib.get('sec-type')],
                "section_id": [sec.attrib.get('id')],
                "section_title": [sectionTitle],
                "sub_section_title": [subSectionTitle],
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



def chunkFigure(sec,section,subsection,metadata,chunkIndex):

    fig=sec.xpath(".//fig")


    figs=[]
    for f in fig:
        #print(f)
        inputText=" ".join(f.xpath(".//text()"))

        fig={
            "PMCID":[metadata["PMCID"]],
            "PMID":[metadata["PMID"]],
            "doi":[metadata["doi"]],
            "journal":[metadata["journal"]],
            "year":[metadata["year"]],
            "topic":[metadata["topic"]],
            "article-title": [metadata["article-title"]],
            "chunk_type": ["fig"],#[sec.attrib.get('sec-type')],
            "section_id": [f.attrib.get('id')],
            "section_title": [section],
            "sub_section_title": [subsection],
            "section_path": [None],
            "chunk_text":[inputText],
            "section_chunk_id": [None],#[indexOfSectionChunk],
            "chunk_index": [chunkIndex], 
            "token_count": [len(inputText)]
        }
        figs.append(pd.DataFrame(fig))
    return figs




def getBlankDataframe():
    return pd.DataFrame(columns=["article-title","PMCID","PMID","doi","journal","year","topic","chunk_type","section_id","section_title","section_path","chunk_text", "section_chunk_id","chunk_index","token_count"])

if __name__ == '__main__':

    source=getDataframe("./pmc_chunker/out/manifest_4000.csv",False)
    #source=getDataframe("./pmc_chunker/data/xml/",True)
    #source=getDataframe("./pmc_chunker/data/xml2/",True)



    #parseXMLSection(1000,4000,340)

    parseXMLSectionParagraph(source,100000,1000,300)

    #chk=chunker.getFixedChunker(700)
    chk=chunker.getModel("sentence-transformers/all-MiniLM-L6-v2",700)
    #parseXMLSectionParagraphModel(source,200000,1000,10000,chk)



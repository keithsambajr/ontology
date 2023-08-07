Title: Embeddings of entity relationship prediction 
Project description: A knowledge graph (KG) is a graph-structured data model used to store interlinked descriptions of entities (objects, events, situations or abstract concepts) that also encoding the semantics underlying the terminology used to represent entities and their relations. In knowledge graphs, nodes represent entities within a domain (either concepts or individuals) and edges represent the relations between the nodes. The schema of a KG is often called an ontology.

Ontology embeddings can make information in ontologies available as background knowledge within machine learning models. Ontologies can be used as structured output in those tasks that aim to predict whether some entity has a relation with one or more ontology classes, such as predicting gene–disease or drug–disease associations (using disease ontologies as output).

As a minimum requirements, these tasks need to satisfy the hierarchical constraints imposed by the ontologies in the output space, for instance: if an entity e is predicted to be associated with a class C, and that class Cis a subclass of D, then e must also be associated with D.

There are at least five different approaches to using hierarchical relationships as constraints in classification models: flat, local per node, local per parent, local per level, and global hierarchical classification [1].

The aim of this project is to implement (in Python) and compare these different approaches in some use cases defined as part of the project.

Implementation:
3 algorithms : flat classification(used as baseline approach)
              Local per parent node
              Global/Big bang classification



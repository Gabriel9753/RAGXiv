from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

DATABASE_URL = "postgresql://admin:admin@localhost:5432/metadata_db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Association table for many-to-many relationship between papers and authors
author_paper = Table(
    "author_paper",
    Base.metadata,
    Column("author_id", String, ForeignKey("authors.author_id"), primary_key=True),
    Column("paper_id", String, ForeignKey("papers.arxiv_id"), primary_key=True),
)


class PaperMetadata(Base):
    __tablename__ = "papers"
    id = Column(Integer, primary_key=True, index=True)
    arxiv_id = Column(String, unique=True)
    semantic_scholar_id = Column(String, unique=True)
    title = Column(String)
    super_category = Column(String)
    update_year = Column(Integer)
    reference_count = Column(Integer)
    citation_count = Column(Integer)
    author_count = Column(Integer)
    # Relationship to authors (many-to-many)
    authors = relationship("Author", secondary=author_paper, back_populates="papers")
    # Self-referencing many-to-many for references
    references = relationship("Reference", back_populates="paper")


class Author(Base):
    __tablename__ = "authors"
    id = Column(Integer, primary_key=True, index=True)
    author_id = Column(String, unique=True)
    name = Column(String)
    papers = relationship("PaperMetadata", secondary=author_paper, back_populates="authors")


class Reference(Base):
    __tablename__ = "references"
    id = Column(Integer, primary_key=True, index=True)
    arxiv_id = Column(String, ForeignKey("papers.arxiv_id"))
    cited_semantic_scholar_id = Column(String)
    title = Column(String)
    paper = relationship("PaperMetadata", back_populates="references")


def init_db():
    # Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)


class DBManager:
    def __init__(self):
        self.session = SessionLocal()

    def insert_paper(
        self,
        arxiv_id,
        semantic_scholar_id,
        title,
        super_category,
        update_year,
        reference_count,
        citation_count,
        author_count,
        authors,
        references,
    ):
        # first check if the paper already exists
        paper = self.session.query(PaperMetadata).filter(PaperMetadata.arxiv_id == arxiv_id).first()
        if paper:
            return

        new_paper = PaperMetadata(
            arxiv_id=arxiv_id,
            semantic_scholar_id=semantic_scholar_id,
            title=title,
            super_category=super_category,
            update_year=update_year,
            reference_count=reference_count,
            citation_count=citation_count,
            author_count=author_count,
        )
        # Add authors
        if len(authors) > 0:
            for author_name in authors:
                _id, name = author_name["authorId"], author_name["name"]
                if _id is None or name is None:
                    continue
                _id, name = str(_id), str(name)
                author = self.session.query(Author).filter(Author.author_id == _id).first()
                if not author:
                    author = Author(author_id=_id, name=name)
                new_paper.authors.append(author)

        # # Add references
        if len(references) > 0:
            for reference in references:
                source_arxiv_id = arxiv_id
                paper_semantic_scholar_id, reference_title = reference["paperId"], reference["title"]
                if reference_title is None or paper_semantic_scholar_id is None:
                    continue
                paper_semantic_scholar_id, reference_title = str(paper_semantic_scholar_id), str(reference_title)

                referenced_paper = (
                    self.session.query(PaperMetadata)
                    .filter(PaperMetadata.semantic_scholar_id == paper_semantic_scholar_id)
                    .first()
                )

                if not referenced_paper:
                    referenced_paper = PaperMetadata(
                        semantic_scholar_id=paper_semantic_scholar_id, title=reference_title
                    )

                new_reference = Reference(
                    arxiv_id=source_arxiv_id, cited_semantic_scholar_id=paper_semantic_scholar_id, title=reference_title
                )
                new_paper.references.append(new_reference)

        self.session.add(new_paper)
        self.session.commit()

    def get_metadata_from_arxivid(self, arxiv_id):
        paper = self.session.query(PaperMetadata).filter(PaperMetadata.arxiv_id == arxiv_id).first()
        if paper:
            return {
                'arxiv_id': paper.arxiv_id,
                'semantic_scholar_id': paper.semantic_scholar_id,
                'title': paper.title,
                'super_category': paper.super_category,
                'update_year': paper.update_year,
                'reference_count': paper.reference_count,
                'citation_count': paper.citation_count,
                'author_count': paper.author_count
            }
        return None

    def get_authors_from_arxivid(self, arxiv_id):
        paper = self.session.query(PaperMetadata).filter(PaperMetadata.arxiv_id == arxiv_id).first()
        if paper:
            return [{'author_id': author.author_id, 'name': author.name} for author in paper.authors]
        return []

    def get_references_from_arxivid(self, arxiv_id):
        paper = self.session.query(PaperMetadata).filter(PaperMetadata.arxiv_id == arxiv_id).first()
        if paper:
            return [
                {
                    'semantic_scholar_id': ref.cited_semantic_scholar_id,
                    'title': ref.title
                }
                for ref in paper.references
            ]
        return []

    def close(self):
        self.session.close()

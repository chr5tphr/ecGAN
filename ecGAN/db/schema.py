from sqlalchemy import Column, ForeignKey, Integer, String, DateTime, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

#%%
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from datetime import datetime
from contextlib import contextmanager
engine = create_engine('postgresql://python@localhost/python')

Session = sessionmaker()
sess = Session(bind=engine.connect())

Base.metadata.create_all(engine)

sess.query(Setup).all()

setup = Setup(
    time_created=datetime.now(),
    time_modified=datetime.now(),
    config={'yes':'no'},
    )

setup2 = Setup(
    id=None,
    time_created=datetime.now(),
    time_modified=datetime.now(),
    config={'yes':'no'},
    )

setup2.id

sess.add(setup2)
sess.flush()
sess.rollback()
snap.id

snap = Snapshot(time_created=datetime.now(),epoch=0,setup=setup)

sess.add_all([setup,snap])

sess.commit()

sess.close()
#%%


Base = declarative_base()

class Setup(Base):
    __tablename__ = 'setup'

    id = Column(Integer, primary_key=True, autoincrement=True)
    time_created = Column(DateTime)
    time_modified = Column(DateTime)
    storage_path = Column(String)
    config = Column(JSON)

    snapshots = relationship("Snapshot", back_populates='setup')

class Snapshot(Base):
    __tablename__ = 'snapshot'

    id = Column(Integer, primary_key=True, autoincrement=True)
    setup_id = Column(Integer, ForeignKey('setup.id'), unique=True)
    time_created = Column(DateTime)
    epoch = Column(Integer)

    setup = relationship("Setup", back_populates='snapshots')
    network_instances = relationship("SnapshotNetworkInstance", back_populates='snapshot')

class Network(Base):
    __tablename__ = 'network'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    architecture = Column(String)

    network_instances = relationship("NetworkInstance", back_populates='network')

class NetworkInstance(Base):
    __tablename__ = 'network_instance'

    id = Column(Integer, primary_key=True, autoincrement=True)
    network_id = Column(Integer, ForeignKey('network.id'), unique=True)
    param = Column(String)

    network = relationship("Network", back_populates='network_instances')
    snapshots = relationship("SnapshotNetworkInstance", back_populates='network_instance')
    explanations = relationship("Explanation", back_populates='network_instance')
    sample_data = relationship("ProcessedData", back_populates='network_instance')

class SnapshotNetworkInstance(Base):
    __tablename__ = 'snapshot_network_instance'

    snapshot_id = Column(Integer, ForeignKey('snapshot.id'), primary_key=True)
    network_instance_id = Column(Integer, ForeignKey('network_instance.id'), primary_key=True, unique=True)
    type = Column(String)

    snapshot = relationship("Snapshot", back_populates='network_instances')
    network_instance = relationship("NetworkInstance", back_populates='snapshots')

class Data(Base):
    __tablename__ = 'data'

    id = Column(Integer, primary_key=True, autoincrement=True)
    time_created = Column(DateTime)
    data = Column(String)

class ProcessedData(Base):
    __tablename__ = 'processed_data'

    id = Column(Integer, primary_key=True, autoincrement=True)
    network_instance_id = Column(Integer, ForeignKey('network_instance.id'), unique=True)
    input_id = Column(Integer, ForeignKey('data.id'), unique=True)
    output_id = Column(Integer, ForeignKey('data.id'), unique=True)

    network_instance = relationship("NetworkInstance", back_populates='sample_data')
    input_data = relationship("Data", foreign_keys=[input_id])
    output_data = relationship("Data", foreign_keys=[output_id])
    explanations = relationship("Explanation", back_populates='processed_data')

class Explanation(Base):
    __tablename__ = 'explanation'

    id = Column(Integer, primary_key=True, autoincrement=True)
    network_instance_id = Column(Integer, ForeignKey('network_instance.id'), unique=True)
    processed_data_id = Column(Integer, ForeignKey('processed_data.id'), unique=True)
    data_id = Column(Integer, ForeignKey('data.id'), unique=True)
    method = Column(String)

    processed_data = relationship("ProcessedData", back_populates='explanations')
    network_instance = relationship("NetworkInstance", back_populates='explanations')
    data = relationship("Data")

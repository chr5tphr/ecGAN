from sqlalchemy import Column, Integer, String, LargeBinary, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.ex.declarative import declarative_base

Base = declarative_base()

class Setup(Base):
    __tablename__ = 'setup'

    id = Column(Integer, primary_key=True, autoincrement=True)
    time_created = Column(DateTime)
    time_modified = Column(DateTime)
    config = Column(JSON)

    snapshots = relationship("Snapshot", back_populates='setup')

class Snapshot(Base):
    __tablename__ = 'snapshot'

    id = Column(Integer, primary_key=True, autoincrement=True)
    setup_id = Column(Integer, ForeignKey('setup.id'))
    time_created = Column(DateTime)
    epoch = Column(Integer)

    setup = relationship("Setup", back_populates='snapshot')
    network_instances = relationship("SnapshotNetworkInstance", back_populates='snapshot')
    generated_samples = relationship("ProcessedData", back_populates='snapshot')

class Network(Base):
    __tablename__ = 'network'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    architecture = Column(String)

    network_instances = relationship("NetworkInstance", back_populates='network')

class NetworkInstance(Base):
    __tablename__ = 'network_instance'

    id = Column(Integer, primary_key=True, autoincrement=True)
    network_id = Column(Integer, ForeignKey('network.id'))
    param = Column(LargeBinary)

    network = relationship("Network", back_populates='network_instance')
    snapshots = relationship("SnapshotNetworkInstance", back_populates='network_instance')
    explanations = relationship("Explanation", back_populates='network_instance')

class SnapshotNetworkInstance(Base):
    __tablename__ = 'snapshot_network_instance'

    snapshot_id = Column(Integer, ForeignKey('snapshot.id'), primary_key=True)
    network_instance_id = Column(Integer, ForeignKey('network_instance.id'), primary_key=True)
    type = Column(String)

    snapshot = relationship("Snapshot", back_populates='network_instance')
    network_instance = relationship("NetworkInstance", back_populates='snapshot')

class ProcessedData(Base):
    __tablename__ = 'processed_data'

    id = Column(Integer, primary_key=True, autoincrement=True)
    network_instance_id = Column(Integer, ForeignKey('network_instance.id'))
    time_created = Column(DateTime)
    input = Column(LargeBinary)
    output = Column(LargeBinary)

    network_instance = relationship("Snapshot", back_populates='processed_data')
    explanations = relationship("Explanation", back_populates='processed_data')

class Explanation(Base):
    __tablename__ = 'explanation'

    id = Column(Integer, primary_key=True, autoincrement=True)
    network_instance_id = Column(Integer, ForeignKey('network_instance.id'))
    processed_data_id = Column(Integer, ForeignKey('processed_data.id'))
    time_created = Column(DateTime)
    method = Column(String)
    data = Column(LargeBinary)

    processed_data = relationship("ProcessedData", back_populates='explanation')
    network_instance = relationship("NetworkInstance", back_populates='explanation')

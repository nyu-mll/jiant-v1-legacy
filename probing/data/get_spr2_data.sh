#!/bin/bash

TARGET_DIR=$1

pushd $TARGET_DIR

set -e

# wget http://decomp.net/wp-content/uploads/2015/08/protoroles_eng_pb.tar.gz
# tar zxvf protorole_eng_pb.tar.gz

# wget http://decomp.net/wp-content/uploads/2015/08/protoroles_eng_ud1.2.tar.gz
# tar zxvf protoroles_eng_ud1.2.tar.gz

# Univeral Dependencies 1.2 for English Web Treebank (ewt) source text.
wget https://github.com/UniversalDependencies/UD_English/archive/r1.2.tar.gz
mkdir ud
tar -zxvf r1.2.tar.gz -C ud

# Semantic Proto Roles annotations.
wget http://decomp.io/projects/semantic-proto-roles/protoroles_eng_udewt.tar.gz
mkdir protoroles
tar -xvzf protoroles_eng_udewt.tar.gz -C protoroles

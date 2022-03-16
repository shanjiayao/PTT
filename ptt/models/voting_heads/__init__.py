# -*- coding: utf-8 -*-
from .voting_head_template import VotingHeadTemplate
from .centroids_voting_head import CentroidVotingHead
from .box_voting_head import BoxVotingHead

__all__ = {
    'VotingHeadTemplate': VotingHeadTemplate,
    'CentroidVotingHead': CentroidVotingHead,
    'BoxVotingHead': BoxVotingHead,
}
use postgres go;

-- Sistema de Inteligencia Prescritiva para a Industria Farmaceutica
-- Modelo relacional e views analiticas para PostgreSQL/Supabase.
--
-- Convencoes:
--   * periodo representa sempre o primeiro dia do mes;
--   * valores monetarios usam numeric(18,2);
--   * quantidades e indices usam numeric para evitar imprecisao de float;
--   * metricas derivadas, como market share e cobertura, sao calculadas em views;
--   * os nomes de produtos, setores e outlets devem ser sinteticos.
begin;

-- ============================================================================
-- DIMENSOES
-- ============================================================================
drop table if exists public.dim_produto cascade;

create table public.dim_produto (
  id_produto bigint generated always as identity primary key,
  nome_sintetico varchar(120) not null,
  tipo_produto varchar(80), -- CONCORRENTE / PRÓPRIO
  constraint ck_dim_produto_nome check (btrim(nome_sintetico) <> '')
);

comment on table public.dim_produto is 'Cadastro de produtos sinteticos. Nao deve conter o nome confidencial do produto real.';

drop table if exists public.dim_mercado cascade;

create table public.dim_mercado (
  id_mercado bigint generated always as identity primary key,
  nome_sintetico varchar(120) not null,
  constraint ck_dim_mercado_nome check (btrim(nome_sintetico) <> '')
);

comment on table public.dim_mercado is 'Mercados ou classes terapeuticas utilizados nas analises.';

drop table if exists public.produto_mercado cascade;

create table public.produto_mercado (
  id_produto bigint not null,
  id_mercado bigint not null,
  constraint pk_produto_mercado primary key (id_produto, id_mercado),
  constraint fk_produto_mercado_produto foreign key (id_produto) references public.dim_produto (id_produto) on update cascade on delete restrict,
  constraint fk_produto_mercado_mercado foreign key (id_mercado) references public.dim_mercado (id_mercado) on update cascade on delete restrict
);

comment on table public.produto_mercado is 'Relacionamento historico entre produtos e mercados.';

drop table if exists public.dim_setor cascade;

create table public.dim_setor (
  id_setor bigint generated always as identity primary key,
  codigo_setor varchar(30) not null unique,
  nome_sintetico varchar(120) not null,
  regiao varchar(30),
  uf char(2),
  constraint ck_dim_setor_codigo check (btrim(codigo_setor) <> ''),
  constraint ck_dim_setor_nome check (btrim(nome_sintetico) <> ''),
  constraint ck_dim_setor_uf check (
    uf is null
    or uf ~ '^[A-Z]{2}$'
  ) --Letras maiúsculas de A-Z e apenas 2 letras
);

comment on table public.dim_setor is 'Territorios ou setores comerciais sinteticos.';

drop table if exists public.dim_outlet cascade;

create table public.dim_outlet (
  id_outlet bigint generated always as identity primary key,
  codigo_outlet varchar(14) not null unique, --ALOCACAO
  id_setor bigint not null, -- RESPONSAVEL PELA ALOCACAO
  tipo_outlet varchar(10), -- CNPJ OU BRICK
  uf char(2) not null,
  cidade varchar(250),
  bairro varchar(250),
  cep varchar(8),
  canal varchar(80),
  subcanal varchar(80),
  constraint fk_dim_outlet_setor foreign key (id_setor) references public.dim_setor (id_setor) on update cascade on delete restrict,
  constraint ck_dim_outlet_codigo check (btrim(codigo_outlet) <> ''),
  constraint ck_dim_outlet_uf check (uf ~ '^[A-Z]{2}$')
);

comment on table public.dim_outlet is 'PDVs, bricks ou agrupamentos geograficos sinteticos. Nao armazena CNPJ ou endereco.';

-- ============================================================================
-- FATOS
-- ============================================================================
--OBJETIVO - USARA PARA CALCULAR O DESEMPENHO SEMANAL DO COLABORADOR
create table public.fato_objetivo_mensal (
  id_fato bigint generated always as identity primary key,
  periodo date not null,
  id_produto bigint not null,
  id_mercado bigint not null,
  id_setor bigint not null,
  objetivo_unidades float not null,
  objetivo_valorado float not null,
  constraint uq_fato_objetivo_grao unique (periodo, id_produto, id_mercado, id_setor),
  constraint fk_fato_mercado_produto foreign key (id_produto) references public.dim_produto (id_produto) on update cascade on delete restrict,
  constraint fk_fato_mercado_mercado foreign key (id_mercado) references public.dim_mercado (id_mercado) on update cascade on delete restrict,
  constraint fk_fato_mercado_setor foreign key (id_setor) references public.dim_setor (id_setor) on update cascade on delete restrict,
  constraint ck_fato_mercado_periodo check (periodo = date_trunc('month', periodo)::date)
);

--TABELA COM A DEMANDA DO COLABORADOR MENSAL E DO MERCADO - SERÁ UTILIZADA PARA CALCULAR O MKT SHARE
create table public.fato_mercado_mensal (
  id_fato bigint generated always as identity primary key,
  periodo date not null,
  id_produto bigint not null,
  id_mercado bigint not null,
  id_setor bigint not null,
  id_outlet bigint not null,
  preco_medio float,
  demanda_mercado_und float not null,
  demanda_produto_und float not null,
  demanda_mercado_val float not null,
  demanda_produto_val float not null,
  constraint uq_fato_mercado_mensal_grao unique (
    periodo,
    id_produto,
    id_mercado,
    id_setor,
    id_outlet
  ),
  constraint fk_fato_mercado_produto foreign key (id_produto) references public.dim_produto (id_produto) on update cascade on delete restrict,
  constraint fk_fato_mercado_mercado foreign key (id_mercado) references public.dim_mercado (id_mercado) on update cascade on delete restrict,
  constraint fk_fato_mercado_setor foreign key (id_setor) references public.dim_setor (id_setor) on update cascade on delete restrict,
  constraint fk_fato_mercado_outlet foreign key (id_outlet) references public.dim_outlet (id_outlet) on update cascade on delete restrict,
  constraint ck_fato_mercado_periodo check (periodo = date_trunc('month', periodo)::date),
  constraint ck_fato_mercado_valores check (
    demanda_mercado_und >= 0
    and demanda_produto_und >= 0
    and demanda_produto_und <= demanda_mercado_und
  )
);

comment on table public.fato_mercado_mensal is 'Fato principal no grao mes, produto, mercado, setor e outlet.';

--TABELA COM A DEMANDA DO COLABORADOR SEMANAL E DO MERCADO - SERÁ UTILIZADA PARA CALCULAR O MKT SHARE
create table public.fato_mercado_semanal (
  id_fato bigint generated always as identity primary key,
  periodo date not null,
  inicio_semana date not null,
  fim_semana date not null,
  id_produto bigint not null,
  id_mercado bigint not null,
  id_setor bigint not null,
  id_outlet bigint not null,
  preco_medio float,
  demanda_mercado_und float not null,
  demanda_produto_und float not null,
  demanda_mercado_val float not null,
  demanda_produto_val float not null,
  constraint uq_fato_mercado_semanal_grao unique (
    periodo,
    id_produto,
    id_mercado,
    id_setor,
    id_outlet
  ),
  constraint fk_fato_mercado_produto foreign key (id_produto) references public.dim_produto (id_produto) on update cascade on delete restrict,
  constraint fk_fato_mercado_mercado foreign key (id_mercado) references public.dim_mercado (id_mercado) on update cascade on delete restrict,
  constraint fk_fato_mercado_setor foreign key (id_setor) references public.dim_setor (id_setor) on update cascade on delete restrict,
  constraint fk_fato_mercado_outlet foreign key (id_outlet) references public.dim_outlet (id_outlet) on update cascade on delete restrict,
  constraint ck_fato_mercado_periodo check (periodo = date_trunc('month', periodo)::date),
  constraint ck_fato_mercado_valores check (
    demanda_mercado_und >= 0
    and demanda_produto_und >= 0
    and demanda_produto_und <= demanda_mercado_und
  )
);
comment on table public.fato_mercado_semanal is 'Fato principal no grao semanal, produto, mercado, setor e outlet.';

--TABELA DE NOTICIAS
create table public.noticia (
  id_noticia bigint generated always as identity primary key,
  data_publicacao timestamptz not null,
  fonte varchar(120) not null,
  titulo text not null,
  descricao text,
  url text not null unique,
  idioma varchar(10) not null default 'pt-BR',
  categoria varchar(80),
  sentimento varchar(10),
  score_sentimento numeric(6, 5),
  nivel_impacto varchar(10),
  confianca_classificacao numeric(6, 5),
  id_produto bigint,
  id_mercado bigint,
  processada_em timestamptz,
  criado_em timestamptz not null default now(),
  constraint fk_noticia_produto foreign key (id_produto) references public.dim_produto (id_produto) on update cascade on delete set null,
  constraint fk_noticia_mercado foreign key (id_mercado) references public.dim_mercado (id_mercado) on update cascade on delete set null,
  constraint ck_noticia_fonte check (btrim(fonte) <> ''),
  constraint ck_noticia_titulo check (btrim(titulo) <> ''),
  constraint ck_noticia_url check (btrim(url) <> ''),
  constraint ck_noticia_sentimento check (
    sentimento is null
    or sentimento in ('negativo', 'neutro', 'positivo')
  ),
  constraint ck_noticia_score check (
    score_sentimento is null
    or score_sentimento between -1 and 1
  ),
  constraint ck_noticia_impacto check (
    nivel_impacto is null
    or nivel_impacto in ('baixo', 'medio', 'alto')
  ),
  constraint ck_noticia_confianca check (
    confianca_classificacao is null
    or confianca_classificacao between 0 and 1
  )
);

comment on table public.noticia is 'Noticias publicas enriquecidas com sentimento, impacto e entidades associadas.';

-- ============================================================================
-- INDICES
-- ============================================================================
create index ix_produto_mercado_mercado on public.produto_mercado (id_mercado, id_produto);

create index ix_dim_outlet_setor on public.dim_outlet (id_setor);

create index ix_dim_outlet_alocacao on public.dim_outlet (codigo_outlet,id_setor,tipo_outlet,uf);

create index ix_fato_mercado_periodo on public.fato_mercado_mensal (periodo);

create index ix_fato_mercado_serie on public.fato_mercado_mensal (id_produto, id_mercado, id_outlet, periodo);

create index ix_fato_mercado_setor_periodo on public.fato_mercado_mensal (id_setor, periodo);

create index ix_fato_desempenho_inicio on public.fato_mercado_semanal (inicio_semana);

create index ix_fato_desempenho_serie on public.fato_mercado_semanal (id_produto, id_outlet, inicio_semana);

create index ix_noticia_publicacao on public.noticia (data_publicacao);

create index ix_noticia_produto on public.noticia (id_produto, data_publicacao);

create index ix_noticia_mercado on public.noticia (id_mercado, data_publicacao);

create index ix_noticia_busca_textual on public.noticia using gin (
  to_tsvector(
    'portuguese',
    coalesce(titulo, '') || ' ' || coalesce(descricao, '')
  )
);

-- ============================================================================
-- VIEWS ANALITICAS
-- ============================================================================
create or replace view public.vw_cota_mensal_fechada
with
  (security_invoker = true) as
select
  f.periodo,
  f.id_produto,
  p.codigo_produto,
  p.nome_sintetico as produto,
  f.id_mercado,
  m.codigo_mercado,
  m.nome_sintetico as mercado,
  f.id_setor,
  s.codigo_setor,
  s.nome_sintetico as setor,
  sum(f.objetivo) as objetivo,
  sum(f.cota) as cota
from
  public.fato_mercado_mensal f
  join public.dim_produto p using (id_produto)
  join public.dim_mercado m using (id_mercado)
  join public.dim_setor s using (id_setor)
group by
  f.periodo,
  f.id_produto,
  p.codigo_produto,
  p.nome_sintetico,
  f.id_mercado,
  m.codigo_mercado,
  m.nome_sintetico,
  f.id_setor,
  s.codigo_setor,
  s.nome_sintetico;

comment on view public.vw_cota_mensal_fechada is 'Objetivo e cota consolidados no mes por produto, mercado e setor.';

create or replace view public.vw_demanda_mensal
with
  (security_invoker = true) as
select
  date_trunc('month', d.inicio_semana)::date as periodo,
  d.id_produto,
  p.codigo_produto,
  d.id_setor,
  s.codigo_setor,
  d.id_outlet,
  o.codigo_outlet,
  sum(d.cota_semanal) as cota_mensal,
  sum(d.demanda_realizada) as demanda_realizada,
  sum(d.unidades_faturadas) as unidades_faturadas,
  sum(d.valor_realizado) as valor_realizado,
  sum(d.demanda_realizada) / nullif(sum(d.cota_semanal), 0) as cobertura
from
  public.fato_desempenho_semanal d
  join public.dim_produto p using (id_produto)
  join public.dim_setor s using (id_setor)
  join public.dim_outlet o using (id_outlet)
group by
  date_trunc('month', d.inicio_semana)::date,
  d.id_produto,
  p.codigo_produto,
  d.id_setor,
  s.codigo_setor,
  d.id_outlet,
  o.codigo_outlet;

comment on view public.vw_demanda_mensal is 'Demanda e cobertura mensais obtidas a partir do fato semanal.';

create or replace view public.vw_market_share
with
  (security_invoker = true) as
select
  f.id_fato,
  f.periodo,
  f.id_produto,
  p.codigo_produto,
  f.id_mercado,
  m.codigo_mercado,
  f.id_setor,
  s.codigo_setor,
  f.id_outlet,
  o.codigo_outlet,
  o.uf,
  f.demanda_produto,
  f.demanda_mercado,
  f.demanda_produto / nullif(f.demanda_mercado, 0) as market_share_outlet,
  sum(f.demanda_produto) over (
    partition by
      f.periodo,
      f.id_produto,
      f.id_mercado,
      o.uf
  ) / nullif(
    sum(f.demanda_mercado) over (
      partition by
        f.periodo,
        f.id_produto,
        f.id_mercado,
        o.uf
    ),
    0
  ) as market_share_uf,
  sum(f.demanda_produto) over (
    partition by
      f.periodo,
      f.id_produto,
      f.id_mercado
  ) / nullif(
    sum(f.demanda_mercado) over (
      partition by
        f.periodo,
        f.id_produto,
        f.id_mercado
    ),
    0
  ) as market_share_brasil,
  f.demanda_produto / nullif(
    sum(f.demanda_produto) over (
      partition by
        f.periodo,
        f.id_produto,
        f.id_mercado
    ),
    0
  ) as participacao_produto,
  f.demanda_mercado / nullif(
    sum(f.demanda_mercado) over (
      partition by
        f.periodo,
        f.id_produto,
        f.id_mercado
    ),
    0
  ) as participacao_mercado
from
  public.fato_mercado_mensal f
  join public.dim_produto p using (id_produto)
  join public.dim_mercado m using (id_mercado)
  join public.dim_setor s using (id_setor)
  join public.dim_outlet o using (id_outlet);

comment on view public.vw_market_share is 'Market share por outlet, UF e Brasil, calculado sem duplicar metricas na tabela fato.';

create or replace view public.vw_cobertura_cota
with
  (security_invoker = true) as
select
  d.id_desempenho,
  d.inicio_semana,
  d.fim_semana,
  d.id_produto,
  p.codigo_produto,
  d.id_setor,
  s.codigo_setor,
  d.id_outlet,
  o.codigo_outlet,
  d.cota_semanal,
  d.demanda_realizada,
  d.demanda_realizada / nullif(d.cota_semanal, 0) as cobertura,
  case
    when d.cota_semanal = 0 then 'sem_cota'
    when d.demanda_realizada >= d.cota_semanal then 'atingida'
    when d.demanda_realizada >= d.cota_semanal * 0.90 then 'atencao'
    else 'critica'
  end as status_cota,
  d.data_disponibilizacao
from
  public.fato_desempenho_semanal d
  join public.dim_produto p using (id_produto)
  join public.dim_setor s using (id_setor)
  join public.dim_outlet o using (id_outlet);

comment on view public.vw_cobertura_cota is 'Cobertura semanal e classificacao operacional do atingimento da cota.';

create or replace view public.vw_dataset_previsao_demanda
with
  (security_invoker = true) as
with
  serie as (
    select
      f.*,
      lag(f.demanda_produto, 1) over w as demanda_lag_1,
      lag(f.demanda_produto, 2) over w as demanda_lag_2,
      lag(f.demanda_produto, 3) over w as demanda_lag_3,
      avg(f.demanda_produto) over (
        partition by
          f.id_produto,
          f.id_mercado,
          f.id_outlet
        order by
          f.periodo rows between 3 preceding
          and 1 preceding
      ) as media_movel_3m,
      lead(f.demanda_produto, 1) over w as demanda_alvo_proximo_mes
    from
      public.fato_mercado_mensal f
    window
      w as (
        partition by
          f.id_produto,
          f.id_mercado,
          f.id_outlet
        order by
          f.periodo
      )
  )
select
  s.periodo,
  s.id_produto,
  s.id_mercado,
  s.id_setor,
  s.id_outlet,
  o.uf,
  o.canal,
  o.subcanal,
  o.tipo_outlet,
  o.potencial,
  s.objetivo,
  s.cota,
  s.preco_medio,
  s.investimento_promocional,
  s.estoque_disponivel,
  s.indice_sazonal,
  s.evento_mercado,
  s.demanda_lag_1,
  s.demanda_lag_2,
  s.demanda_lag_3,
  s.media_movel_3m,
  coalesce(n.quantidade_noticias, 0) as quantidade_noticias,
  coalesce(n.sentimento_medio, 0) as sentimento_medio,
  coalesce(n.noticias_alto_impacto, 0) as noticias_alto_impacto,
  s.demanda_alvo_proximo_mes,
  s.data_disponibilizacao
from
  serie s
  join public.dim_outlet o using (id_outlet)
  left join lateral (
    select
      count(*) as quantidade_noticias,
      avg(noticia.score_sentimento) as sentimento_medio,
      count(*) filter (
        where
          noticia.nivel_impacto = 'alto'
      ) as noticias_alto_impacto
    from
      public.noticia
    where
      noticia.processada_em is not null
      and date_trunc('month', noticia.data_publicacao)::date = s.periodo
      and (
        noticia.id_produto is null
        or noticia.id_produto = s.id_produto
      )
      and (
        noticia.id_mercado is null
        or noticia.id_mercado = s.id_mercado
      )
  ) n on true;

comment on view public.vw_dataset_previsao_demanda is 'Dataset temporal para prever a demanda do proximo mes. A coluna demanda_alvo_proximo_mes e o target.';

create or replace view public.vw_dataset_risco_cota
with
  (security_invoker = true) as
with
  serie as (
    select
      d.*,
      d.demanda_realizada / nullif(d.cota_semanal, 0) as cobertura_atual,
      lag(
        d.demanda_realizada / nullif(d.cota_semanal, 0),
        1
      ) over w as cobertura_lag_1,
      lag(d.demanda_realizada, 1) over w as demanda_lag_1,
      avg(d.demanda_realizada) over (
        partition by
          d.id_produto,
          d.id_outlet
        order by
          d.inicio_semana rows between 4 preceding
          and 1 preceding
      ) as media_demanda_4s,
      lead(d.demanda_realizada, 1) over w as demanda_alvo_proxima_semana,
      lead(
        d.demanda_realizada / nullif(d.cota_semanal, 0),
        1
      ) over w as cobertura_alvo_proxima_semana
    from
      public.fato_desempenho_semanal d
    window
      w as (
        partition by
          d.id_produto,
          d.id_outlet
        order by
          d.inicio_semana
      )
  )
select
  s.inicio_semana,
  s.fim_semana,
  s.id_produto,
  s.id_setor,
  s.id_outlet,
  o.uf,
  o.canal,
  o.potencial,
  s.cota_semanal,
  s.demanda_realizada,
  s.unidades_faturadas,
  s.estoque_disponivel,
  s.dias_uteis,
  s.cobertura_atual,
  s.cobertura_lag_1,
  s.demanda_lag_1,
  s.media_demanda_4s,
  s.demanda_alvo_proxima_semana,
  s.cobertura_alvo_proxima_semana,
  case
    when s.cobertura_alvo_proxima_semana is null then null
    when s.cobertura_alvo_proxima_semana < 1 then 1
    else 0
  end as alvo_nao_atingir_cota,
  s.data_disponibilizacao
from
  serie s
  join public.dim_outlet o using (id_outlet);

comment on view public.vw_dataset_risco_cota is 'Dataset para classificar o risco de nao atingir a cota na semana seguinte.';

create or replace view public.vw_dataset_prescritivo
with
  (security_invoker = true) as
with
  base as (
    select
      f.*,
      avg(f.demanda_produto) over (
        partition by
          f.id_produto,
          f.id_mercado,
          f.id_outlet
        order by
          f.periodo rows between 3 preceding
          and 1 preceding
      ) as demanda_base_3m,
      sum(f.objetivo) over (
        partition by
          f.periodo,
          f.id_produto,
          f.id_mercado
      ) as objetivo_total,
      sum(f.demanda_produto) over (
        partition by
          f.periodo,
          f.id_produto,
          f.id_mercado
      ) as demanda_total
    from
      public.fato_mercado_mensal f
  )
select
  b.periodo,
  b.id_produto,
  b.id_mercado,
  b.id_setor,
  b.id_outlet,
  o.uf,
  o.canal,
  o.potencial,
  b.demanda_produto,
  b.demanda_base_3m,
  b.cota as cota_atual,
  b.objetivo,
  b.estoque_disponivel,
  b.investimento_promocional,
  b.demanda_produto / nullif(b.demanda_mercado, 0) as market_share,
  greatest(
    0,
    coalesce(b.demanda_base_3m, b.demanda_produto) - b.cota
  ) as oportunidade_incremental,
  case
    when b.demanda_total = 0 then 0
    else b.objetivo_total * b.demanda_produto / b.demanda_total
  end as cota_sugerida_baseline,
  case
    when b.estoque_disponivel is null then null
    else least(
      b.estoque_disponivel,
      case
        when b.demanda_total = 0 then 0
        else b.objetivo_total * b.demanda_produto / b.demanda_total
      end
    )
  end as cota_sugerida_limitada_estoque,
  b.data_disponibilizacao
from
  base b
  join public.dim_outlet o using (id_outlet);

comment on view public.vw_dataset_prescritivo is 'Baseline explicavel para redistribuicao de cotas; a recomendacao final deve usar previsoes e restricoes do otimizador.';

-- ============================================================================
-- SEGURANCA BASICA PARA O SUPABASE
-- ============================================================================
-- RLS fica habilitado nas tabelas expostas pelo schema public.
-- A carga de dados e o treinamento devem usar um backend seguro com service_role.
-- Usuarios autenticados recebem somente leitura; nenhuma tabela e liberada para anon.
alter table public.dim_produto enable row level security;

alter table public.dim_mercado enable row level security;

alter table public.produto_mercado enable row level security;

alter table public.dim_setor enable row level security;

alter table public.dim_outlet enable row level security;

alter table public.fato_mercado_mensal enable row level security;

alter table public.fato_desempenho_semanal enable row level security;

alter table public.noticia enable row level security;

grant
select
  on table public.dim_produto,
  public.dim_mercado,
  public.produto_mercado,
  public.dim_setor,
  public.dim_outlet,
  public.fato_mercado_mensal,
  public.fato_desempenho_semanal,
  public.noticia to authenticated;

grant
select
  on table public.vw_cota_mensal_fechada,
  public.vw_demanda_mensal,
  public.vw_market_share,
  public.vw_cobertura_cota,
  public.vw_dataset_previsao_demanda,
  public.vw_dataset_risco_cota,
  public.vw_dataset_prescritivo to authenticated;

revoke all on table public.dim_produto,
public.dim_mercado,
public.produto_mercado,
public.dim_setor,
public.dim_outlet,
public.fato_mercado_mensal,
public.fato_desempenho_semanal,
public.noticia,
public.vw_cota_mensal_fechada,
public.vw_demanda_mensal,
public.vw_market_share,
public.vw_cobertura_cota,
public.vw_dataset_previsao_demanda,
public.vw_dataset_risco_cota,
public.vw_dataset_prescritivo
from
  anon;

create policy "Leitura autenticada de produtos" on public.dim_produto for
select
  to authenticated using (true);

create policy "Leitura autenticada de mercados" on public.dim_mercado for
select
  to authenticated using (true);

create policy "Leitura autenticada de produto mercado" on public.produto_mercado for
select
  to authenticated using (true);

create policy "Leitura autenticada de setores" on public.dim_setor for
select
  to authenticated using (true);

create policy "Leitura autenticada de outlets" on public.dim_outlet for
select
  to authenticated using (true);

create policy "Leitura autenticada do mercado mensal" on public.fato_mercado_mensal for
select
  to authenticated using (true);

create policy "Leitura autenticada do desempenho semanal" on public.fato_desempenho_semanal for
select
  to authenticated using (true);

create policy "Leitura autenticada de noticias" on public.noticia for
select
  to authenticated using (true);

commit;